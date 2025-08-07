import { FileText, Brain, CheckCircle } from "lucide-react";

export default function HowItWorks() {
  const steps = [
    {
      icon: FileText,
      title: "أدخل النص",
      description: "الصق النص الإخباري الذي تريد تحليل مصداقيته",
    },
    {
      icon: Brain,
      title: "التحليل بالذكاء الاصطناعي",
      description: "يقوم نموذج AraBERT المدرب بتحليل النص وتقييم مصداقيته",
    },
    {
      icon: CheckCircle,
      title: "احصل على النتيجة",
      description: "اطلع على تقييم مفصل مع نسبة الثقة والاحتماليات",
    },
  ];

  return (
    <section className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748] mb-4">
            كيف تعمل الأداة؟
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto text-lg">
            عملية بسيطة من ثلاث خطوات للحصول على تحليل دقيق لمصداقية الأخبار
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
          {steps.map((step, index) => (
            <div key={index} className="text-center group">
              <div className="relative mb-6">
                <div className="w-20 h-20 bg-gradient-to-br from-[#799EFF] to-[#6B8EFF] rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg transition-transform duration-300 group-hover:scale-110">
                  <step.icon className="w-10 h-10 text-white" />
                </div>
                <div className="w-8 h-8 bg-[#2D3748] text-white rounded-full flex items-center justify-center mx-auto -mt-2 text-sm font-bold shadow-md">
                  {index + 1}
                </div>
              </div>
              <h3 className="text-xl font-semibold text-[#2D3748] mb-3">
                {step.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
