import { Card, CardContent } from "@/components/ui/card";
import {
  ExternalLink,
  Github,
  Linkedin,
  Bot,
  Microscope,
  BarChart3,
  Cpu,
} from "lucide-react";

export default function About() {
  return (
    <section
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gray-50"
      id="about"
    >
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748] mb-4">
            عن المشروع
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            نظام شامل للكشف عن الأخبار المزيفة في المحتوى العربي
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Feature Cards */}
          <Card className="border-gray-200 shadow-lg hover:shadow-xl transition-shadow duration-300 bg-white">
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="font-semibold text-[#2D3748] mb-3">
                نهج مزدوج النماذج
              </h3>
              <ul className="text-gray-600 space-y-2 text-sm">
                <li>• Traditional ML: XGBoost</li>
                <li>• Transformers: AraBERT</li>
                <li>• Accuracy: 93.48%</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-gray-200 shadow-lg hover:shadow-xl transition-shadow duration-300 bg-white">
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Microscope className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="font-semibold text-[#2D3748] mb-3">
                خط إنتاج احترافي
              </h3>
              <ul className="text-gray-600 space-y-2 text-sm">
                <li>• التحقق المتقاطع 5 مرات</li>
                <li>• معمارية نظيفة ومعيارية</li>
                <li>• جاهز للإنتاج الفوري</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-gray-200 shadow-lg hover:shadow-xl transition-shadow duration-300 bg-white">
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Cpu className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="font-semibold text-[#2D3748] mb-3">
                تقنيات متقدمة
              </h3>
              <ul className="text-gray-600 space-y-2 text-sm">
                <li>• معالجة النصوص العربية</li>
                <li>• تحليل المشاعر والسياق</li>
                <li>• نشر سحابي متطور</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Performance Comparison */}
        <Card className="border-gray-200 shadow-lg bg-white">
          <CardContent className="p-8">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-semibold text-[#2D3748] mb-2">
                الأداء المقارن للنماذج
              </h3>
              <p className="text-gray-600">
                مقارنة شاملة بين نماذج التعلم الآلي المختلفة
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-emerald-50 p-6 rounded-xl border border-emerald-200">
                <div className="flex items-center gap-3 mb-4">
                  <BarChart3 className="w-6 h-6 text-emerald-600" />
                  <h4 className="font-semibold text-emerald-800">
                    AraBERT (Transformers)
                  </h4>
                </div>
                <ul className="text-emerald-700 space-y-2">
                  <li>• Accuracy: 93.48%</li>
                  <li>• AUC: 98.1%</li>
                  <li>• F1-Score: 93.53%</li>
                </ul>
              </div>

              <div className="bg-blue-50 p-6 rounded-xl border border-blue-200">
                <div className="flex items-center gap-3 mb-4">
                  <BarChart3 className="w-6 h-6 text-blue-600" />
                  <h4 className="font-semibold text-blue-800">
                    XGBoost (Machine Learning)
                  </h4>
                </div>
                <ul className="text-blue-700 space-y-2">
                  <li>• Accuracy: 90.61%</li>
                  <li>• AUC: 96.53%</li>
                  <li>• F1-Score: 90.79%</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
