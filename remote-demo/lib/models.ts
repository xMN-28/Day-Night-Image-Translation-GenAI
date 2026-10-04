export type Direction = "day_to_night" | "night_to_day";

export type ModelOption = {
  id: string;
  label: string;
  note: string;
  directions: Direction[];
  recommended?: boolean;
  experimental?: boolean;
};

export const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "v2_best",
    label: "LumiCycle V2 — best (4.5k)",
    note: "Recommended checkpoint; strongest locally trained result.",
    directions: ["day_to_night", "night_to_day"],
    recommended: true,
  },
  {
    id: "v2_final",
    label: "LumiCycle V2 — final (12k)",
    note: "Later V2 checkpoint for comparison.",
    directions: ["day_to_night", "night_to_day"],
  },
  {
    id: "v1_best",
    label: "LumiCycle V1 — best (13k)",
    note: "Original best checkpoint.",
    directions: ["day_to_night", "night_to_day"],
  },
  {
    id: "v1_final",
    label: "LumiCycle V1 — final (40k)",
    note: "Long-run V1 checkpoint; did not improve over 13k.",
    directions: ["day_to_night", "night_to_day"],
  },
  {
    id: "v2_1_best",
    label: "V2.1 ablation — best (5.5k)",
    note: "Failed detail-preservation ablation; retained for honest comparison.",
    directions: ["day_to_night", "night_to_day"],
    experimental: true,
  },
  {
    id: "v2_1_final",
    label: "V2.1 ablation — final (6k)",
    note: "Failed V2.1 final checkpoint.",
    directions: ["day_to_night", "night_to_day"],
    experimental: true,
  },
  {
    id: "lumirender_final",
    label: "LumiRender — final (34k)",
    note: "Physics-guided experiment; mainly learned global darkening.",
    directions: ["day_to_night"],
    experimental: true,
  },
];

export const MODEL_IDS = new Set(MODEL_OPTIONS.map((model) => model.id));

export function supportsDirection(modelId: string, direction: Direction) {
  return MODEL_OPTIONS.some(
    (model) => model.id === modelId && model.directions.includes(direction),
  );
}
