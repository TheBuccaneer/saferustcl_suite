# run_baseline.ps1
# Läuft Baseline-Matrix (t={2,4,8} × conflict={low,med,high}) ops-basiert,
# startet jeden Lauf mit /high + /affinity und legt Artefakte pro Lauf ab.

param(
  [int[]]   $Threads   = @(2,4,8),
  [string[]]$Conflicts = @('low','med','high'),
  [long]    $Ops       = 1000000,
  [long]    $Seed      = 1,
  [string]  $OutRoot   = "results\FGOS\seed1"
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

foreach ($c in $Conflicts) {
  foreach ($t in $Threads) {
    $dest = Join-Path $OutRoot "$c\t$t"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    # Affinity-Maske für Kerne 0..(t-1)
    $mask = 0; for ($i=0; $i -lt $t; $i++) { $mask = $mask -bor (1 -shl $i) }
    $maskHex = ('0x{0:X}' -f $mask)

    # Cargo-Lauf (hohe Priorität + Affinität), im Workspace-Root
    $args = @(
      '/c','start','""','/wait','/b','/high','/affinity', $maskHex,
      'cargo','run','-p','hpc-core','--release','--features','memtrace','--example','stm_abort','--',
      '--threads',"$t",'--conflict',"$c",'--ops',"$Ops",'--seed',"$Seed"
    )
    Start-Process -FilePath 'cmd.exe' -ArgumentList $args -WorkingDirectory $Root -NoNewWindow -Wait

    # Artefakte einsammeln
    $abort = Join-Path $Root 'memtrace_abort.csv'
    $sum   = Join-Path $Root 'memtrace_summary.txt'
    if (!(Test-Path $abort) -or !(Test-Path $sum)) { throw "Artefakte fehlen für t=$t, c=$c." }

    Move-Item -LiteralPath $abort -Destination (Join-Path $dest 'memtrace_abort.csv') -Force
    Move-Item -LiteralPath $sum   -Destination (Join-Path $dest 'memtrace_summary.txt') -Force

    $events = Join-Path $Root 'memtrace.csv'
if (Test-Path $events) {
  Move-Item -LiteralPath $events -Destination (Join-Path $dest 'memtrace_events.csv') -Force
}
  }
}

# Kurz-Summary schreiben
Get-ChildItem $OutRoot -Recurse -Filter memtrace_summary.txt |
  ForEach-Object {
    $ev=(Select-String -Path $_.FullName -Pattern 'events_total:\s*(\d+)').Matches[0].Groups[1].Value
    $ab=(Select-String -Path $_.FullName -Pattern 'aborts:\s*(\d+)').Matches[0].Groups[1].Value
    "$($_.DirectoryName)`t$ev`t$ab"
  } | Set-Content (Join-Path $OutRoot 'summary_matrix.tsv')


# --- robuste Rate-Tabelle ---
$OPS = 1000000
$rows = @()

Get-ChildItem $OutRoot -Recurse -Filter memtrace_summary.txt | ForEach-Object {
  $leaf = Split-Path $_.DirectoryName -Leaf
  if ($leaf -notmatch '^t(\d+)$') { return }      # nur Verzeichnisse t2,t4,t8,...
  $t = [int]$Matches[1]
  $c = Split-Path (Split-Path $_.DirectoryName -Parent) -Leaf
  $ev = [double]((Select-String $_.FullName -Pattern 'events_total:\s*(\d+)').Matches[0].Groups[1].Value)
  $ab = [double]((Select-String $_.FullName -Pattern 'aborts:\s*(\d+)').Matches[0].Groups[1].Value)
  $rows += [pscustomobject]@{ conflict=$c; threads=$t; events=$ev; aborts=$ab; rate=($ab/$OPS) }
}

$rows | Sort-Object conflict,threads |
  ForEach-Object { "{0}`t{1}`t{2}`t{3}`t{4:N6}" -f $_.conflict,$_.threads,$_.events,$_.aborts,$_.rate } |
  Set-Content (Join-Path $OutRoot 'summary_rates.tsv')