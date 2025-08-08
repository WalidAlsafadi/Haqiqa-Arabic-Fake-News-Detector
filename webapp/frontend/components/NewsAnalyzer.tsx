"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Loader2, AlertTriangle } from "lucide-react";

interface AnalysisResult {
  prediction: string;
  confidence: number;
  Real: number;
  Fake: number;
  error?: string;
}

// Backend connection function for AraBERT Hugging Face Space
async function analyzeNews(text: string): Promise<AnalysisResult> {
  try {
    const spaceUrl =
      "https://walidalsafadi-arabert-fake-news-detector.hf.space";

    // Step 1: Submit the request
    const submitResponse = await fetch(`${spaceUrl}/call/predict_news`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [text] }),
    });

    if (!submitResponse.ok) {
      const errorText = await submitResponse.text();
      return {
        prediction: "",
        confidence: 0,
        Real: 0,
        Fake: 0,
        error: errorText,
      };
    }

    const submitResult = await submitResponse.json();
    if (!submitResult.event_id) {
      return {
        prediction: "",
        confidence: 0,
        Real: 0,
        Fake: 0,
        error: "No event_id received from backend.",
      };
    }

    // Step 2: Get the result (SSE streaming)
    const resultResponse = await fetch(
      `${spaceUrl}/call/predict_news/${submitResult.event_id}`
    );
    if (!resultResponse.ok || !resultResponse.body) {
      const errorText = await resultResponse.text();
      return {
        prediction: "",
        confidence: 0,
        Real: 0,
        Fake: 0,
        error: errorText,
      };
    }

    const reader = resultResponse.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (Array.isArray(data) && data.length > 0) {
              const output = data[0];
              // Parse AraBERT markdown response
              const predictionLine = output.match(/التنبؤ:\*\* ([^\n]+)/);
              const confidenceLine = output.match(/الثقة:\*\* ([\d.]+)%/);
              const realLine = output.match(
                /احتمالية الخبر الحقيقي: ([\d.]+)%/
              );
              const fakeLine = output.match(/احتمالية الخبر المزيف: ([\d.]+)%/);

              return {
                prediction: predictionLine
                  ? predictionLine[1].trim()
                  : "غير محدد",
                confidence: confidenceLine ? parseFloat(confidenceLine[1]) : 0,
                Real: realLine ? parseFloat(realLine[1]) : 0,
                Fake: fakeLine ? parseFloat(fakeLine[1]) : 0,
              };
            }
          } catch (e) {
            // Ignore parsing errors for SSE chunks
          }
        }
      }
    }
    return {
      prediction: "",
      confidence: 0,
      Real: 0,
      Fake: 0,
      error: "No valid result received.",
    };
  } catch (error: any) {
    return {
      prediction: "",
      confidence: 0,
      Real: 0,
      Fake: 0,
      error: error.message || "Unknown error",
    };
  }
}

export default function NewsAnalyzer() {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");

  const analyzeText = async () => {
    if (!text.trim()) {
      setError("يرجى إدخال نص للتحليل");
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setResult(null);

    try {
      const analysisResult = await analyzeNews(text);

      if (analysisResult.error) {
        setError(analysisResult.error);
      } else {
        setResult(analysisResult);
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      setError("حدث خطأ أثناء التحليل. يرجى المحاولة مرة أخرى.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section
      id="analyzer"
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 scroll-mt-16"
    >
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748] mb-4">
            محلل الأخبار
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto text-lg">
            الصق النص الإخباري في المربع أدناه وسنقوم بتحليل مصداقيته باستخدام
            نموذج الذكاء الاصطناعي المتقدم
          </p>
        </div>

        {/* Single Large Container */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
          <div className="grid lg:grid-cols-2 gap-0 min-h-[500px]">
            {/* Input Section - Right Side */}
            <div className="p-8 border-l border-gray-200">
              <div className="space-y-6 h-full flex flex-col">
                <label className="block text-base font-medium text-[#2D3748]">
                  النص الإخباري
                </label>
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="الصق النص الإخباري هنا للتحليل..."
                  className="flex-1 min-h-[280px] text-right resize-none border-gray-200 focus:border-[#799EFF] focus:ring-2 focus:ring-[#799EFF]/20 rounded-xl text-base leading-relaxed p-4"
                  dir="rtl"
                />
                <Button
                  onClick={analyzeText}
                  disabled={!text.trim() || isAnalyzing}
                  className="w-full bg-[#799EFF] hover:bg-[#6B8EFF] disabled:bg-gray-300 text-white py-4 rounded-xl transition-all duration-300 font-medium text-lg shadow-lg hover:shadow-xl hover:scale-[1.02] transform"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      جاري التحليل...
                    </>
                  ) : (
                    "تحليل النص"
                  )}
                </Button>
              </div>
            </div>

            {/* Results Section - Left Side */}
            <div className="p-8 bg-gray-50">
              {result ? (
                <div className="h-full flex flex-col justify-center">
                  <div className="text-center mb-6">
                    <Badge
                      className={`text-lg px-6 py-3 mb-6 rounded-full font-medium ${
                        result.prediction === "حقيقي"
                          ? "bg-emerald-100 text-emerald-800 hover:bg-emerald-100"
                          : "bg-red-100 text-red-800 hover:bg-red-100"
                      }`}
                    >
                      {result.prediction === "حقيقي" ? "خبر حقيقي" : "خبر مضلل"}
                    </Badge>
                  </div>

                  <div className="flex justify-center mb-8">
                    <div className="w-40 h-40">
                      <CircularProgressbar
                        value={result.confidence}
                        text={`${result.confidence}%`}
                        styles={buildStyles({
                          textSize: "14px",
                          pathColor:
                            result.prediction === "حقيقي"
                              ? "#10B981"
                              : "#EF4444",
                          textColor: "#2D3748",
                          trailColor: "#F3F4F6",
                          pathTransitionDuration: 0.8,
                        })}
                      />
                    </div>
                  </div>

                  <div className="space-y-4 mb-6">
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                      <span className="text-gray-700 font-medium">
                        احتمالية الصدق:
                      </span>
                      <span className="font-semibold text-emerald-600 text-lg">
                        {result.Real}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                      <span className="text-gray-700 font-medium">
                        احتمالية التضليل:
                      </span>
                      <span className="font-semibold text-red-600 text-lg">
                        {result.Fake}%
                      </span>
                    </div>
                  </div>

                  <div className="border-t pt-6 space-y-3">
                    <p className="text-sm text-gray-600 text-center">
                      <strong>إصدار النموذج:</strong> Fine-tuned AraBERT V1
                      (BETA)
                    </p>
                    <div className="flex items-start gap-3 text-sm text-amber-700 bg-amber-50 p-4 rounded-lg">
                      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>
                        تعتمد هذه الأداة على الذكاء الاصطناعي وقد تخطئ في بعض
                        الحالات.
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col justify-center items-center text-center">
                  <div className="text-gray-400 mb-6">
                    <div className="w-20 h-20 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
                      <AlertTriangle className="w-10 h-10" />
                    </div>
                  </div>
                  <p className="text-gray-500 text-lg">
                    أدخل النص الإخباري واضغط على &ldquo;تحليل النص&rdquo; لرؤية
                    النتائج
                  </p>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border-t border-red-200">
              <div className="flex items-center gap-3 text-red-700 justify-center">
                <AlertTriangle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
