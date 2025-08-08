"use client";

import { Button } from "@/components/ui/button";
import { ArrowDown, Info } from "lucide-react";
import Image from "next/image";

export default function Hero() {
  const scrollToAnalyzer = () => {
    document.getElementById("analyzer")?.scrollIntoView({ behavior: "smooth" });
  };

  const scrollToAbout = () => {
    document.getElementById("about")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="py-20 sm:py-32 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-white via-gray-50/30 to-gray-50">
      <div className="max-w-6xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Brand Illustration - Left Side */}
          <div className="flex justify-center lg:justify-start order-1 lg:order-1">
            <div className="w-80 h-80 sm:w-96 sm:h-96 flex items-center justify-center">
              <Image
                src="/hero-illustration.svg"
                alt="حقيقة - كاشف الأخبار المزيفة"
                width={384}
                height={384}
                className="w-full h-full object-contain"
                priority
              />
            </div>
          </div>

          {/* Content - Right Side */}
          <div className="text-center lg:text-right order-2 lg:order-2">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-[#2D3748] mb-6 leading-tight tracking-tight">
              اكتشف الحقيقة في الأخبار العربية
            </h1>
            <p className="text-lg sm:text-xl text-gray-600 mb-10 leading-relaxed">
              أداة ذكية تستخدم الذكاء الاصطناعي لتحليل مصداقية الأخبار العربية
              وتساعدك على التمييز بين الحقيقة والأخبار المضللة
            </p>

            {/* Dual Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Button
                onClick={scrollToAnalyzer}
                className="bg-[#799EFF] hover:bg-[#6B8EFF] text-white px-8 py-4 text-lg font-medium rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105 transform group"
              >
                جرّب الأداة الآن
                <ArrowDown className="mr-2 h-5 w-5 transition-transform duration-300 group-hover:translate-y-1" />
              </Button>
              <Button
                onClick={scrollToAbout}
                variant="outline"
                className="border-[#799EFF] text-[#799EFF] hover:bg-[#799EFF] hover:text-white px-8 py-4 text-lg font-medium rounded-xl transition-all duration-300 shadow-md hover:shadow-lg"
              >
                عن المشروع
                <Info className="mr-2 h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
