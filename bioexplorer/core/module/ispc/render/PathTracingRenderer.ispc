/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "PathTracingRenderer.ih"

#include "utils/BioExplorerRandomGenerator.ih"
#include "utils/BioExplorerSkyBox.ih"

uniform const uint16 NB_MAX_REBOUNDS = 20;
uniform const float skypower = 3.0f;
uniform const float skypower_zerobounce = 1.0f;

/**
    Returns the refracted vector according to the direction of the incident ray,
   he normal to the surface, and localRefraction indices
    @param direction Direction of the incident ray
    @param normal Normal to the surface
    @param n1 In localRefraction index
    @param n2 Out localRefraction index
    @return Refracted vector
*/
inline vec3f refractedVector(const varying vec3f& direction,
                             const varying vec3f& normal,
                             const varying float n1, const varying float n2)
{
    if (n2 == 0.f)
        return direction;
    const float eta = n1 / n2;
    const float cos1 = -dot(direction, normal);
    const float cos2 = 1.f - eta * eta * (1.f - cos1 * cos1);
    if (cos2 > 0.f)
        return eta * direction + (eta * cos1 - sqrt(cos2)) * normal;
    return direction;
}

inline vec3f pathTracingContribution(
    const uniform PathTracingRenderer* uniform self,
    varying ScreenSample& sample, const varying vec3f& sunDirection)
{
    vec3f Kd;
    Ray localray = sample.ray;
    vec3f accucolor = make_vec3f(0.f);
    vec3f mask = make_vec3f(1.f);

    // path tracing loop
    for (int bounces = 0; bounces < 3; bounces++)
    {
        traceRay(self->super.model, localray);

        // if ray misses scene (no hit occurs), return background colour
        if (localray.geomID < 0)
        {
            localray.t = inf;
            const vec3f bgcol =
                make_vec3f(skyboxMapping((Renderer*)self, localray,
                                         self->bgMaterial)) *
                skypower;
            accucolor = accucolor + mask * bgcol;
            break;
        }
        DifferentialGeometry dg;
        postIntersect(self->super.model, dg, localray,
                      DG_NG | DG_NS | DG_NORMALIZE | DG_FACEFORWARD |
                          DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

        uniform Material* material = dg.material;
        uniform AdvancedMaterial* objMaterial =
            (uniform AdvancedMaterial*)material;
        if (!objMaterial)
            Kd = make_vec3f(dg.color);
        else
            foreach_unique(mat in objMaterial) Kd =
                mat->Kd * make_vec3f(dg.color);

        // compute local orthornormal base at hitpoint
        const vec3f N = dg.Ns;

        // compute ray direction of uniformly random diffuse ray
        const vec3f raydir =
            getRandomVector(self->useHardwareRandomizer, self->super.fb->size.x,
                            sample, N, self->randomNumber);

        // origin of new ray in path is hitpoint of previous ray in path
        const vec3f hitpoint = dg.P + dg.epsilon * N;
        setRay(localray, hitpoint, raydir);
        localray.t0 = dg.epsilon;
        localray.t = self->aoDistance;

        //***************************************
        //           direct lighting
        //***************************************

        // cosine weighted contribution
        const float sunLight = dot(dg.Ns, sunDirection);

        if (sunLight > 0.f) // if surface is facing light
        {
            // create and trace a shadowray towards the sun
            Ray sunRay;
            setRay(sunRay, hitpoint, sunDirection);
        }

        // take surface color at this bounce's hitpoint into account
        mask = mask * Kd;

    } // end bounces
    return accucolor;
}

/**
    Renderer a pixel color according to a given location in the screen space.
    @param self Pointer to current renderer
    @param sample Screen sample containing information about the ray, and the
           location in the screen space.
*/
inline vec3f PathTracingRenderer_shadeRay(
    const uniform PathTracingRenderer* uniform self,
    varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    vec3f color = make_vec3f(0.f);

    const uniform Light* uniform light = self->lights[0];
    const vec2f s = make_vec2f(0.5f);
    DifferentialGeometry dg;
    const varying Light_SampleRes lightSample = light->sample(light, dg, s);
    const vec3f sunDirection = lightSample.dir;
    const vec3f radiance = lightSample.weight;

    float pathOpacity = 1.f;
    float oldlocalRefraction = 1.f;
    int depth = 0;

    sample.z = inf;
    sample.alpha = 0.f;

    bool moreRebounds = true;
    while (moreRebounds && depth < NB_MAX_REBOUNDS && pathOpacity > 0.f)
    {
        float opacity = 1.f;
        float reflection = 0.f;
        float refraction = 1.f;

        traceRay(self->super.model, ray);

        if (ray.geomID < 0)
        {
            // No intersection. Return skybox color
            ray.t = inf;
            color = self->showBackground
                        ? make_vec3f(skyboxMapping((Renderer*)self, ray,
                                                   self->bgMaterial)) *
                              skypower_zerobounce
                        : make_vec3f(0.f);

            // No Geometry intersection. No need to iterate more
            moreRebounds = false;
        }
        else
        {
            // else, we've got a hit!
            DifferentialGeometry dg;
            postIntersect(self->super.model, dg, ray,
                          DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD |
                              DG_MATERIALID | DG_COLOR | DG_TEXCOORD);
            const vec3f intersection = dg.P;

            uniform Material* material = dg.material;
            uniform AdvancedMaterial* objMaterial =
                (uniform AdvancedMaterial*)material;

            vec3f normal = dg.Ns;
            vec3f Kd = make_vec3f(0.f);
            vec3f Ks = make_vec3f(0.f);
            float Ns = 0.f;

            if (!objMaterial)
            {
                Kd = make_vec3f(dg.color);
                opacity = dg.color.w;
            }
            else
                foreach_unique(mat in objMaterial)
                {
                    opacity = mat->d;

                    // Diffuse
                    Kd = mat->Kd;
                    if (valid(mat->map_Kd))
                    {
                        const vec4f value = get4f(mat->map_Kd, dg);
                        Kd = make_vec3f(value);
                        opacity *= value.w;
                    }

                    // Specular
                    Ks = mat->Ks;
                    if (valid(mat->map_Ks))
                        Ks = get3f(mat->map_Ks, dg);
                    Ns = mat->Ns;
                    if (valid(mat->map_Ns))
                        Ns = get1f(mat->map_Ns, dg);

                    // Refraction
                    refraction = mat->refraction;

                    // Reflection
                    reflection = mat->reflection;
                    if (valid(mat->map_reflection))
                    {
                        const vec4f value = get4f(mat->map_reflection, dg);
                        Ns = value.x * 100.f;
                        reflection = value.w;
                    }

                    // Normal mapping
                    if (valid(mat->map_Bump))
                        normal = normalize(normal * get3f(mat->map_Bump, dg));

                    // Emissive mapping
                    if (valid(mat->map_a))
                        Kd = Kd + get3f(mat->map_a, dg);
                }

            const vec3f reflectedNormal =
                ray.dir - 2.f * dot(ray.dir, normal) * normal;

            // Phong + Blinn shading
            float cosNL = max(0.f, dot(sunDirection, normal));
            const float cosLR = max(0.f, dot(sunDirection, reflectedNormal));
            const vec3f specularColor = Ks * powf(cosLR, Ns);

            // Path tracing contribution
            const float ao = self->aoStrength;
            const vec3f pathTracingColor =
                ao > 0.f
                    ? Kd * ao *
                          pathTracingContribution(self, sample, sunDirection)
                    : Kd * radiance * cosNL + specularColor;

            // Alpha and Z-Depth
            if (depth == 0)
            {
                sample.z = ray.t;
                sample.alpha = opacity;
            }

            color =
                pathTracingColor * pathOpacity + color * (1.f - pathOpacity);

            // Prepare ray for next iteration
            bool doRefraction = (opacity < 1.f);
            bool doReflection = (reflection > 0.f);
            if (doRefraction && doReflection)
            {
                // Switch between refraction and reflection
                const bool refract = sample.sampleID.z % 2 == 0;
                doRefraction = refract;
                doReflection = !refract;
            }

            if (doRefraction)
            {
                // Refraction
                vec3f correctedNormal = normal;
                const float angle = dot(ray.dir, correctedNormal);
                if (abs(refraction - oldlocalRefraction) < 0.001f &&
                    angle > 0.f)
                {
                    // Exiting object, back to air (refractionindex = 1)
                    refraction = 1.f;
                    correctedNormal = correctedNormal * -1.f;
                }

                ray.t0 = 0.f;
                ray.dir = refractedVector(ray.dir, correctedNormal,
                                          oldlocalRefraction, refraction);
                ray.org = intersection + dg.epsilon * ray.dir;

                pathOpacity -= (1.f - opacity);
                oldlocalRefraction = refraction;
                ++depth;
            }
            else if (doReflection)
            {
                // Reflection
                ray.t0 = 0.f;
                ray.dir = reflectedNormal;
                ray.org = intersection + dg.epsilon * reflectedNormal;
                pathOpacity -= reflection;
                ++depth;
            }
            else
                moreRebounds = false;

            ray.t = inf;
            ray.primID = -1;
            ray.geomID = -1;
            ray.instID = -1;
        }
    }

    return clamp(color * self->exposure, make_vec3f(0.f), make_vec3f(1.f));
}

void PathTracingRenderer_renderSample(uniform Renderer* uniform _self,
                                      void* uniform perFrameData,
                                      varying ScreenSample& sample)
{
    uniform PathTracingRenderer* uniform self =
        (uniform PathTracingRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = PathTracingRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform PathTracingRenderer_create(void* uniform cppE)
{
    uniform PathTracingRenderer* uniform self =
        uniform new uniform PathTracingRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = PathTracingRenderer_renderSample;
    return self;
}

export void PathTracingRenderer_set(
    void* uniform _self, void* uniform bgMaterial, const uniform uint32 spp,
    void** uniform lights, uniform int32 numLights,
    const uniform float timestamp, const uniform uint32 randomNumber,
    const uniform float exposure, const uniform float aoStrength,
    const uniform float aoDistance, const uniform bool useHardwareRandomizer,
    const uniform bool showBackground)
{
    uniform PathTracingRenderer* uniform self =
        (uniform PathTracingRenderer * uniform) _self;

    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->timestamp = timestamp;
    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    self->aoStrength = aoStrength;
    self->aoDistance = aoDistance;

    self->exposure = exposure;
    self->randomNumber = randomNumber;
    self->useHardwareRandomizer = useHardwareRandomizer;
    self->showBackground = showBackground;
}
