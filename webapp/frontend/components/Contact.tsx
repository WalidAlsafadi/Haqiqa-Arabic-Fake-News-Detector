"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Send, Loader2 } from "lucide-react";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<
    "idle" | "success" | "error"
  >("idle");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus("idle");

    try {
      const response = await fetch("https://formspree.io/f/xdoqzqzq", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          message: formData.message,
          _replyto: formData.email,
        }),
      });

      if (response.ok) {
        setSubmitStatus("success");
        setFormData({ name: "", email: "", message: "" });
      } else {
        setSubmitStatus("error");
      }
    } catch (error) {
      console.error("Form submission error:", error);
      setSubmitStatus("error");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <section
      id="contact"
      className="py-16 sm:py-20 px-4 sm:px-6 lg:px-8 bg-gray-50"
    >
      <div className="max-w-4xl mx-auto">
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
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label
                    htmlFor="name"
                    className="block text-sm font-medium text-gray-700 mb-2"
                  >
                    الاسم
                  </label>
                  <Input
                    id="name"
                    name="name"
                    type="text"
                    required
                    value={formData.name}
                    onChange={handleInputChange}
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
                    البريد الإلكتروني
                  </label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    required
                    value={formData.email}
                    onChange={handleInputChange}
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
                  الرسالة
                </label>
                <Textarea
                  id="message"
                  name="message"
                  required
                  value={formData.message}
                  onChange={handleInputChange}
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
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    جاري الإرسال...
                  </>
                ) : (
                  <>
                    <Send className="mr-2 h-5 w-5" />
                    إرسال
                  </>
                )}
              </Button>

              {submitStatus === "success" && (
                <div className="text-center p-4 bg-green-50 text-green-700 rounded-lg">
                  تم إرسال رسالتك بنجاح! سنتواصل معك قريباً.
                </div>
              )}

              {submitStatus === "error" && (
                <div className="text-center p-4 bg-red-50 text-red-700 rounded-lg">
                  حدث خطأ أثناء إرسال الرسالة. يرجى المحاولة مرة أخرى.
                </div>
              )}
            </form>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
