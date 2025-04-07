
namespace snap {
float UniformCoord(float x, float xmin, float xmax, void*) {
  // linear interp, equally weighted from left (x(xmin)=0.) and right
  // (x(xmax)=1.)
  return xmin + (xmax - xmin) * x;
}
}  // namespace snap
