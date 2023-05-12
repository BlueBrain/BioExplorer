#include "GolgiStyleRenderer.ih"

#include <ospray/SDK/common/Model.ih>

/**
    Renderer a pixel color according to a given location in the screen space.
    @param self Pointer to current renderer
    @param sample Screen sample containing information about the ray, and the
           location in the screen space.
*/
inline vec3f GolgiStyleRenderer_shadeRay(
    const uniform GolgiStyleRenderer* uniform self,
    varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    varying vec3f color = make_vec3f(1.f, 1.f, 1.f);

    sample.z = inf;
    sample.alpha = 0.f;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        return color;

    DifferentialGeometry dg;
    postIntersect(self->super.model, dg, ray,
                  DG_NS | DG_NORMALIZE | DG_FACEFORWARD);

    float cosNL = max(0.f, pow(dot(normalize(sample.ray.dir), -1.f * dg.Ns),
                               self->exponent));

    if (self->inverse)
        cosNL = 1.f - cosNL;
    color = make_vec3f(cosNL, cosNL, cosNL);
    sample.alpha = 1.f - cosNL;

    // Z-Depth
    sample.z = ray.t;

    return color;
}

void GolgiStyleRenderer_renderSample(uniform Renderer* uniform _self,
                                     void* uniform perFrameData,
                                     varying ScreenSample& sample)
{
    uniform GolgiStyleRenderer* uniform self =
        (uniform GolgiStyleRenderer * uniform) _self;
    sample.ray.time = 0.f;
    sample.rgb = GolgiStyleRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform GolgiStyleRenderer_create(void* uniform cppE)
{
    uniform GolgiStyleRenderer* uniform self =
        uniform new uniform GolgiStyleRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = GolgiStyleRenderer_renderSample;
    return self;
}

export void GolgiStyleRenderer_set(void* uniform _self,
                                   void* uniform bgMaterial,
                                   const uniform int& spp,
                                   const uniform float& exponent,
                                   const uniform bool& inverse)
{
    uniform GolgiStyleRenderer* uniform self =
        (uniform GolgiStyleRenderer * uniform) _self;

    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;
    self->super.spp = spp;
    self->exponent = exponent;
    self->inverse = inverse;
}
