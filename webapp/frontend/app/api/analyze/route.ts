import { NextResponse } from "next/server";
import { analyzeWithSpace } from "@/lib/hf";

export const runtime = "nodejs"; // required for @gradio/client

type Body = { text?: string; model?: "arabert" | "xgboost" };

export async function POST(req: Request) {
  try {
    const { text, model }: Body = await req.json();

    if (!text?.trim() || !model) {
      return NextResponse.json(
        { ok: false, error: "Bad payload: { text, model } required" },
        { status: 400 }
      );
    }

    const data = await analyzeWithSpace(text, model);
    return NextResponse.json({ ok: true, data });
  } catch (e: any) {
    return NextResponse.json(
      { ok: false, error: e?.message ?? "Server error" },
      { status: 500 }
    );
  }
}
