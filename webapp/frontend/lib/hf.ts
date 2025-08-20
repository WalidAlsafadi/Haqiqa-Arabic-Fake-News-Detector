// frontend/lib/hf.ts
import { Client } from "@gradio/client";

const SPACE_URL = process.env.HF_SPACE_URL!;

type SpaceOut = {
  model: string;
  prediction: "Real" | "Fake";
  confidence: number;
  real_prob: number;
  fake_prob: number;
};

function coerceNumber(x: unknown): number {
  const n = typeof x === "string" ? Number(x) : (x as number);
  return Number.isFinite(n) ? n : 0;
}

export async function analyzeWithSpace(
  text: string,
  model: "arabert" | "xgboost"
): Promise<SpaceOut> {
  const app = await Client.connect(SPACE_URL);
  const res = await app.predict("/predict", [text, model]);

  // Normalize shape: could be obj, [obj], or stringified JSON
  let raw: any = res.data;

  if (typeof raw === "string") {
    try {
      raw = JSON.parse(raw);
    } catch {
      /* leave as is */
    }
  }
  if (Array.isArray(raw)) {
    raw = raw[0];
    if (typeof raw === "string") {
      try {
        raw = JSON.parse(raw);
      } catch {
        /* leave as is */
      }
    }
  }

  // Fallbacks + number coercion to avoid NaN on the client
  const model_name = (raw?.model ?? model) as string;
  const prediction = (raw?.prediction ?? "Real") as "Real" | "Fake";
  const confidence = coerceNumber(raw?.confidence);
  const real_prob = coerceNumber(raw?.real_prob);
  const fake_prob = coerceNumber(raw?.fake_prob);

  return { model: model_name, prediction, confidence, real_prob, fake_prob };
}
