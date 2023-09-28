#include "GolgiStyleRenderer.ih"

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>

/**
    Renderer a pixel color according to a given location in the screen space.
    @param self Pointer to current renderer
    @param sample Screen sample containing information about the ray, and the
           location in the screen space.
*/
inline vec3f GolgiStyleRenderer_shadeRay(const uniform GolgiStyleRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;

    sample.z = inf;
    sample.alpha = 0.f;

    traceRay(self->super.super.model, ray);

    if (ray.geomID < 0)
    {
        const vec4f bgSample = skyboxMapping((Renderer*)self, ray, self->super.bgMaterial);
        sample.alpha = bgSample.w;
        return make_vec3f(bgSample);
    }

    DifferentialGeometry dg;
    postIntersect(self->super.super.model, dg, ray, DG_NS | DG_NORMALIZE | DG_FACEFORWARD);

    float cosNL = max(0.f, pow(dot(normalize(sample.ray.dir), -1.f * dg.Ns), self->exponent));

    if (self->inverse)
        cosNL = 1.f - cosNL;
    const vec3f color = make_vec3f(cosNL, cosNL, cosNL);
    sample.alpha = 1.f - cosNL;

    // Z-Depth
    sample.z = ray.t;

    return color;
}

void GolgiStyleRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                     varying ScreenSample& sample)
{
    uniform GolgiStyleRenderer* uniform self = (uniform GolgiStyleRenderer * uniform) _self;
    sample.ray.time = self->super.timestamp;
    if (self->super.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->super.anaglyphIpdOffset;
        const vec3f colorLeft = GolgiStyleRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->super.anaglyphIpdOffset;
        const vec3f colorRight = GolgiStyleRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = GolgiStyleRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform GolgiStyleRenderer_create(void* uniform cppE)
{
    uniform GolgiStyleRenderer* uniform self = uniform new uniform GolgiStyleRenderer;
    Renderer_Constructor(&self->super.super, cppE);
    self->super.super.renderSample = GolgiStyleRenderer_renderSample;
    return self;
}

export void GolgiStyleRenderer_set(void* uniform _self, void* uniform bgMaterial, const uniform int& spp,
                                   const uniform float& exponent, const uniform bool& inverse,
                                   const uniform bool& anaglyphEnabled, const uniform vec3f& anaglyphIpdOffset)
{
    uniform GolgiStyleRenderer* uniform self = (uniform GolgiStyleRenderer * uniform) _self;

    self->super.super.spp = spp;

    self->super.anaglyphEnabled = anaglyphEnabled;
    self->super.anaglyphIpdOffset = anaglyphIpdOffset;
    self->super.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->exponent = exponent;
    self->inverse = inverse;
}