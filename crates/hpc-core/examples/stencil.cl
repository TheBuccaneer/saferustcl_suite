// examples/stencil.cl
// 2D Jacobi-Stencil (4-Point), kein Divergence-Branch im Inneren

__kernel void jacobi(
    __global const float* src,
    __global       float* dst,
    const int width,
    const int height
) {
    // Globale Koordinaten
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Rand behandeln
    if (x == 0 || y == 0 || x == width-1 || y == height-1) {
        int idx = y * width + x;
        dst[idx] = src[idx];
        return;
    }

    // Index für Zentrierung
    int idx = y*width + x;

    // 4-Punkt-Stencil
    float center = src[idx];
    float up     = src[idx - width];
    float down   = src[idx + width];
    float left   = src[idx - 1];
    float right  = src[idx + 1];

    dst[idx] = 0.25f * (up + down + left + right);
}
