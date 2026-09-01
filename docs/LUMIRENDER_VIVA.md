# LumiRender viva questions

**Why not continue from V2 weights?**  V2 encodes a direct RGB translation. LumiRender predicts physically meaningful factors with incompatible outputs, so reusing V2 would both constrain the design and make the originality claim unclear.

**What is Gaussian in the project?**  Anisotropic 2D functions model spatial light falloff, Gaussian kernels model bloom, and Gaussian pyramids separate coarse illumination from details. It is not 3D Gaussian splatting.

**Is the renderer fully physically correct?**  No. It is a differentiable image-space approximation with explicit priors. Full global illumination and hidden reflected geometry cannot be recovered from one photo.

**Why multiple seeds?**  Which lamps are on and what lies outside the frame are unknowable from the daytime input. Seeds represent that uncertainty reproducibly.

**How do you prevent invented objects?**  Geometry/semantic consistency losses, confidence-masked correspondence training, a renderer-dominant architecture and a ±0.03 high-frequency residual limit.

**Why convert sRGB to linear?**  Display RGB is gamma encoded. Addition and multiplication of light are meaningful in approximately linear radiance, not directly in sRGB.

**How are reflections generated?**  The emitter field is reflected in screen space, blurred and vertically stretched, then gated by road/glass masks, roughness, wetness and depth.

**What are the external models?**  Depth Anything V2, Mask2Former and RAFT generate offline training labels/alignment; DINO and a detector evaluate preservation; Turbo is an external benchmark. None is a LumiRender inference backbone.

**Why did V2.1 fail?**  Its detail constraints became easier to satisfy by preserving the daytime appearance, so the required global night transformation weakened.

**Can the output be called the true night scene?**  No. It is one physically plausible rendering. The real future lighting state is not observable from a single daytime image.

**What must pass before the demo uses LumiRender?**  The fixed-suite metrics, plausible-light/reflection gates, structure/object retention, sub-two-second latency and at least 70% blinded preference over V2.
