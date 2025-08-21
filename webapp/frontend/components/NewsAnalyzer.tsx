"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Loader2 } from "lucide-react";

interface AnalysisResult {
  prediction: string; // "حقيقي" | "مزيف"
  confidence: number; // 0-100
  Real: number; // 0-100
  Fake: number; // 0-100
  error?: string;
}

interface RunMetadata {
  model: "arabert" | "xgboost";
  charCount: number;
}

// مساعد لتحديد لون المخطط الدائري بناءً على نسبة الصدق
function getConfidenceColor(realPercentage: number): string {
  if (realPercentage >= 75) return "#10B981"; // emerald-500
  if (realPercentage >= 65) return "#84CC16"; // lime-500
  if (realPercentage >= 45) return "#F59E0B"; // amber-500
  if (realPercentage >= 35) return "#FB923C"; // orange-400
  return "#EF4444"; // red-500
}

// أيقونة التحذير المثلثية (SVG مضمن)
const WarningTriangleIcon = () => (
  <svg
    className="w-4 h-4 text-amber-500 flex-shrink-0"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
    />
  </svg>
);

// دالة تحليل الأخبار (بدون تغيير في API)
async function analyzeNews(
  text: string,
  modelName: "arabert" | "xgboost" = "arabert"
): Promise<AnalysisResult> {
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: modelName }),
    });

    const j = await r.json();

    if (!r.ok || !j.ok) {
      throw new Error(j?.error || `HTTP ${r.status}`);
    }

    const data = j.data as {
      prediction: "Real" | "Fake";
      confidence: number;
      real_prob: number;
      fake_prob: number;
    };

    return {
      prediction: data.prediction === "Real" ? "حقيقي" : "مزيف",
      confidence: Math.round(data.confidence * 100),
      Real: Math.round(data.real_prob * 100),
      Fake: Math.round(data.fake_prob * 100),
    };
  } catch (e: any) {
    return {
      prediction: "",
      confidence: 0,
      Real: 0,
      Fake: 0,
      error: e?.message || "Network error occurred.",
    };
  }
}

export default function NewsAnalyzer() {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState<"arabert" | "xgboost">(
    "arabert"
  );
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [runMeta, setRunMeta] = useState<RunMetadata | null>(null);

  // عداد الأحرف
  const characterCount = text.length;

  const analyzeText = async () => {
    if (!text.trim()) {
      setError("يرجى إدخال نص للتحليل");
      return;
    }

    // حفظ حالة التشغيل الحالية
    const currentRunMeta: RunMetadata = {
      model: selectedModel,
      charCount: characterCount,
    };

    setIsAnalyzing(true);
    setError(null);
    // Keep previous results visible while analyzing

    try {
      const analysisResult = await analyzeNews(text, selectedModel);

      if (analysisResult.error) {
        setError(analysisResult.error);
      } else {
        setResult(analysisResult);
        setRunMeta(currentRunMeta); // Freeze the values used for this run
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
        {/* العنوان مع شارة Beta */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748]">
              محلل الأخبار
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full border border-gray-200 text-gray-700 bg-white/60 backdrop-blur">
              Beta
            </span>
          </div>
          <p className="text-gray-600 max-w-2xl mx-auto text-lg">
            الصق النص الإخباري في المربع أدناه وسنقوم بتحليل مصداقيته باستخدام
            نموذج الذكاء الاصطناعي المتقدم
          </p>
        </div>

        {/* الحاوية الرئيسية */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
          <div className="grid lg:grid-cols-2 gap-0 min-h-[500px]">
            {/* قسم الإدخال - الجانب الأيمن */}
            <div className="p-8 order-1 lg:order-1">
              <div className="space-y-6 h-full flex flex-col">
                {/* مربع النص الكبير */}
                <div className="flex-1 flex flex-col">
                  <label className="block text-base font-medium text-[#2D3748] mb-3">
                    النص الإخباري
                  </label>
                  <Textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="الصق النص الإخباري هنا للتحليل..."
                    className="flex-1 min-h-[300px] text-right resize-none border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 rounded-lg text-base leading-relaxed p-4"
                    dir="rtl"
                  />
                </div>

                {/* الصف المضمن: عداد الأحرف + منتقي النموذج */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  {/* عداد الأحرف (يمين) */}
                  <div className="text-sm text-gray-500 text-right">
                    {characterCount} حرف
                  </div>

                  {/* منتقي النموذج (يسار) */}
                  <div className="w-full sm:w-1/2">
                    <select
                      value={selectedModel}
                      onChange={(e) =>
                        setSelectedModel(
                          e.target.value as "arabert" | "xgboost"
                        )
                      }
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 text-sm bg-white text-left"
                      dir="ltr"
                    >
                      <option value="arabert">AraBERT V1</option>
                      <option value="xgboost">XGBoost V1</option>
                    </select>
                  </div>
                </div>

                {/* زر التحليل */}
                <Button
                  onClick={analyzeText}
                  disabled={!text.trim() || isAnalyzing}
                  className="w-full bg-[#799EFF] hover:bg-[#6B8EFF] disabled:bg-gray-400 text-white py-3 rounded-lg transition-colors font-medium"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      جاري التحليل...
                    </>
                  ) : (
                    "تحليل النص"
                  )}
                </Button>
              </div>
            </div>

            {/* قسم النتائج - الجانب الأيسر */}
            <div className="p-8 bg-gray-50 order-2 lg:order-2">
              {result ? (
                <div className="h-full flex flex-col justify-center space-y-6 divide-y divide-gray-100">
                  {/* شارة النتيجة */}
                  <div className="text-center">
                    <Badge
                      className={`inline-flex items-center px-4 py-1.5 rounded-full text-sm font-medium ${
                        result.prediction === "حقيقي"
                          ? "bg-emerald-100 text-emerald-800 hover:bg-emerald-100"
                          : "bg-red-100 text-red-800 hover:bg-red-100"
                      }`}
                    >
                      {result.prediction === "حقيقي" ? "خبر حقيقي" : "خبر مضلل"}
                    </Badge>
                  </div>

                  {/* المخطط الدائري */}
                  <div className="flex justify-center pt-6">
                    <div className="w-32 h-32">
                      <CircularProgressbar
                        value={result.confidence}
                        text={`${result.confidence}%`}
                        styles={buildStyles({
                          textSize: "18px",
                          textColor: "#2D3748",
                          pathColor: getConfidenceColor(result.Real),
                          trailColor: "#F3F4F6",
                          pathTransitionDuration: 0.8,
                        })}
                      />
                    </div>
                  </div>

                  {/* نسب الصدق والتضليل */}
                  <div className="grid grid-cols-2 gap-4 pt-6">
                    <div className="text-center p-4 bg-white rounded-lg">
                      <div className="text-sm font-medium text-gray-600 mb-1">
                        احتمالية الصدق
                      </div>
                      <div className="text-lg font-semibold text-emerald-600">
                        {result.Real}%
                      </div>
                    </div>
                    <div className="text-center p-4 bg-white rounded-lg">
                      <div className="text-sm font-medium text-gray-600 mb-1">
                        احتمالية التضليل
                      </div>
                      <div className="text-lg font-semibold text-red-600">
                        {result.Fake}%
                      </div>
                    </div>
                  </div>

                  {/* عرض النموذج */}
                  <div className="text-center pt-6">
                    <p className="text-sm text-gray-600">
                      <strong>النموذج:</strong>{" "}
                      {runMeta?.model === "arabert"
                        ? "Haqiqa (AraBERT v1)"
                        : "Haqiqa (XGBoost v1)"}
                    </p>
                  </div>

                  {/* إخلاء مسؤولية الذكاء الاصطناعي */}
                  <div className="pt-4">
                    <p className="text-xs text-gray-500 text-center">
                      تعتمد هذه الأداة على الذكاء الاصطناعي وقد تخطئ في بعض
                      الحالات.
                    </p>
                  </div>

                  {/* تحذير الطول (إذا كان النص قصير) */}
                  {runMeta && runMeta.charCount < 100 && (
                    <div className="pt-4">
                      <p className="text-xs text-amber-600 text-center flex items-center justify-center gap-2">
                        <WarningTriangleIcon />
                        <span>يجب إدخال 100 حرف على الأقل لتحليل دقيق.</span>
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-full flex flex-col justify-center items-center text-center">
                  <div className="text-gray-400 mb-6">
                    <div className="w-20 h-20 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
                      <svg
                        className="w-10 h-10"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        />
                      </svg>
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

          {/* رسالة الخطأ */}
          {error && (
            <div className="p-4 bg-red-50 border-t border-red-200">
              <div className="flex items-center gap-3 text-red-700 justify-center">
                <WarningTriangleIcon />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
