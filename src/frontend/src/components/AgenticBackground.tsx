import bioRobotsBg from "../assets/images/3d-rendering-biorobots-concept.jpg";

/** Full-viewport photo backdrop with a light scrim so foreground content stays readable. */
export function AgenticBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
      <div
        className="absolute inset-0 bg-cover bg-no-repeat"
        style={{
          backgroundImage: `url(${bioRobotsBg})`,
          backgroundPosition: "85% 100%",
        }}
      />
      <div
        className="absolute inset-0 bg-gradient-to-b from-slate-950/90 via-slate-950/65 to-slate-950/88"
        style={{
          boxShadow: "inset 0 0 120px rgb(2 6 23 / 0.65)",
        }}
      />
    </div>
  );
}
