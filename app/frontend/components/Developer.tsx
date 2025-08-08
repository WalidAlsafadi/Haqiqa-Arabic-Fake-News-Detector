import { Card, CardContent } from "@/components/ui/card";
import { Github, Linkedin } from "lucide-react";
import Image from "next/image";

export default function Developer() {
  return (
    <section
      id="developer"
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 scroll-mt-16"
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

        <div className="flex flex-col items-center text-center space-y-6">
          {/* Profile Image */}
          <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-gray-100 shadow-lg">
            <Image
              src="/dev-profile.png"
              alt="وليد الصفدي"
              width={128}
              height={128}
              className="w-full h-full object-cover"
            />
          </div>

          {/* Developer Info */}
          <div className="space-y-2">
            <h3 className="text-2xl font-bold text-[#2D3748]">وليد الصفدي</h3>
            <p className="text-lg text-[#799EFF] font-medium">
              طالب علم بيانات و ذكاء اصطناعي
            </p>
            <p className="text-gray-600">الكلية الجامعية للعلوم التطبيقية</p>
          </div>

          {/* Social Links */}
          <div className="flex gap-4 pt-4">
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
      </div>
    </section>
  );
}
