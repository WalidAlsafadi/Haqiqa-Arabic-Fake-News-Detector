"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Send, Mail, Loader2, CheckCircle, AlertCircle } from "lucide-react";

export default function Contact() {
  const [result, setResult] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    setResult(""); // Clear result to avoid showing loading message under the button

    const formData = new FormData(event.target as HTMLFormElement);
    formData.append("access_key", "8ff1ad13-56cc-4c8c-beb8-23a5e1781b77");

    try {
      const response = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult("تم إرسال رسالتك بنجاح! سنتواصل معك قريباً.");
        (event.target as HTMLFormElement).reset();
      } else {
        setResult(
          "حدث خطأ أثناء الإرسال. يرجى استخدام البريد الإلكتروني المباشر."
        );
      }
    } catch (error) {
      console.error("Form submission error:", error);
      setResult(
        "حدث خطأ أثناء الإرسال. يرجى استخدام البريد الإلكتروني المباشر."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleMailtoFallback = () => {
    const mailtoLink = `mailto:walid.k.alsafadi@gmail.com?subject=استفسار حول مشروع حقيقة&body=مرحبا وليد،%0A%0A`;
    window.open(mailtoLink, "_blank");
  };

  return (
    <section
      id="contact"
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-white scroll-mt-16"
    >
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold text-[#2D3748] mb-4">
            تواصل معنا
          </h2>
          <p className="text-gray-600 text-lg">
            لديك اقتراحات أو تريد المساهمة في تطوير المشروع؟
          </p>
        </div>

        <Card className="border-gray-200 shadow-lg bg-white">
          <CardContent className="p-8">
            <form onSubmit={onSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label
                    htmlFor="name"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    الاسم *
                  </label>
                  <Input
                    id="name"
                    name="name"
                    type="text"
                    required
                    className="text-right border-gray-200 focus:border-[#799EFF] focus:ring-2 focus:ring-[#799EFF]/20"
                    placeholder="أدخل اسمك"
                    dir="rtl"
                  />
                </div>
                <div>
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    البريد الإلكتروني *
                  </label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    required
                    className="text-right border-gray-200 focus:border-[#799EFF] focus:ring-2 focus:ring-[#799EFF]/20"
                    placeholder="أدخل بريدك الإلكتروني"
                    dir="rtl"
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="message"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  الرسالة *
                </label>
                <Textarea
                  id="message"
                  name="message"
                  required
                  className="min-h-[120px] text-right resize-none border-gray-200 focus:border-[#799EFF] focus:ring-2 focus:ring-[#799EFF]/20"
                  placeholder="اكتب رسالتك هنا..."
                  dir="rtl"
                />
              </div>

              <Button
                type="submit"
                disabled={isSubmitting}
                className="w-full bg-[#799EFF] hover:bg-[#6B8EFF] disabled:bg-gray-300 text-white py-3 rounded-lg transition-all duration-300 font-medium text-lg"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="ml-2 h-5 w-5 animate-spin" />
                    جاري الإرسال...
                  </>
                ) : (
                  <>
                    <Send className="ml-2 h-5 w-5" />
                    إرسال الرسالة
                  </>
                )}
              </Button>

              {result && (
                <div
                  className={`text-center p-4 rounded-lg ${
                    result.includes("بنجاح")
                      ? "bg-green-50 text-green-700"
                      : result.includes("خطأ")
                      ? "bg-red-50 text-red-700"
                      : "bg-blue-50 text-blue-700"
                  }`}
                >
                  {result.includes("بنجاح") && (
                    <CheckCircle className="inline ml-2 h-4 w-4" />
                  )}
                  {result.includes("خطأ") && (
                    <AlertCircle className="inline ml-2 h-4 w-4" />
                  )}
                  {result}
                </div>
              )}
            </form>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
