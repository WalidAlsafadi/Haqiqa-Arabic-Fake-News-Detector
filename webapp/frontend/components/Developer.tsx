import { Card, CardContent } from "@/components/ui/card";
import { Github, Linkedin } from "lucide-react";
import Image from "next/image";

export default function Developer() {
  return (
    <section
      id="developer"
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-white"
    >
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748] mb-4">
            المطور
          </h2>
          <p className="text-lg text-gray-600">
            تعرف على الشخص وراء هذا المشروع
          </p>
        </div>

        <Card className="border-gray-200 shadow-xl overflow-hidden">
          <CardContent className="p-0">
            <div className="grid md:grid-cols-2 gap-0">
              {/* Content Section - Left */}
              <div className="p-8 order-2 md:order-1">
                <h3 className="text-2xl font-bold text-[#2D3748] mb-2">
                  وليد الصفدي
                </h3>
                <p className="text-lg text-[#799EFF] font-medium mb-2">
                  طالب علم بيانات و ذكاء اصطناعي
                </p>
                <p className="text-gray-600 mb-6">
                  الكلية الجامعية للعلوم التطبيقية
                </p>

                <div className="flex gap-4">
                  <a
                    href="https://www.linkedin.com/in/walidalsafadi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-12 h-12 bg-gray-100 hover:bg-[#799EFF] hover:text-white rounded-full flex items-center justify-center transition-all duration-300 group"
                    title="LinkedIn"
                  >
                    <Linkedin className="w-6 h-6" />
                  </a>

                  <a
                    href="https://github.com/walidalsafadi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-12 h-12 bg-gray-100 hover:bg-[#799EFF] hover:text-white rounded-full flex items-center justify-center transition-all duration-300"
                    title="GitHub"
                  >
                    <Github className="w-6 h-6" />
                  </a>

                  <a
                    href="https://x.com/walidalsafadi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-12 h-12 bg-gray-100 hover:bg-[#799EFF] hover:text-white rounded-full flex items-center justify-center transition-all duration-300"
                    title="X (Twitter)"
                  >
                    <span className="text-lg font-bold">𝕏</span>
                  </a>

                  <a
                    href="https://huggingface.co/walidalsafadi"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-12 h-12 bg-gray-100 hover:bg-[#799EFF] hover:text-white rounded-full flex items-center justify-center transition-all duration-300"
                    title="Hugging Face"
                  >
                    <span className="text-lg">🤗</span>
                  </a>
                </div>
              </div>

              {/* Image Section - Right */}
              <div className="bg-gradient-to-br from-[#799EFF] to-[#6B8EFF] p-8 flex items-center justify-center order-1 md:order-2">
                <div className="w-48 h-48 rounded-full overflow-hidden border-4 border-white shadow-lg">
                  <Image
                    src="/dev-profile.png"
                    alt="وليد الصفدي"
                    width={192}
                    height={192}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
