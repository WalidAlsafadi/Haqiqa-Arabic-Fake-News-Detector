"use client";

import Image from "next/image";
import { useState } from "react";
import { Menu, X } from "lucide-react";

export default function Header() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const scrollToSection = (sectionId: string) => {
    setIsMobileMenuOpen(false);

    // Small delay to ensure mobile menu closes before scrolling
    setTimeout(() => {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }, 100);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <header className="border-b border-gray-100 bg-white/95 backdrop-blur-md sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo - Top Right in RTL - Perfectly Aligned */}
          <button
            onClick={scrollToTop}
            className="flex items-center hover:opacity-80 transition-opacity"
          >
            <Image
              src="/logo-header.svg"
              alt="حقيقة"
              width={56}
              height={56}
              className="w-14 h-14"
              priority
            />
          </button>

          {/* Desktop Navigation - Centered */}
          <nav className="hidden md:flex items-center gap-8 absolute left-1/2 transform -translate-x-1/2">
            <button
              onClick={() => scrollToSection("how-it-works")}
              className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium"
            >
              كيف تعمل الأداة
            </button>
            <button
              onClick={() => scrollToSection("analyzer")}
              className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium"
            >
              محلل الأخبار
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium"
            >
              عن المشروع
            </button>
            <button
              onClick={() => scrollToSection("developer")}
              className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium"
            >
              المطور
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium"
            >
              تواصل معنا
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {isMobileMenuOpen ? (
              <X className="w-6 h-6 text-gray-600" />
            ) : (
              <Menu className="w-6 h-6 text-gray-600" />
            )}
          </button>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-gray-100">
            <nav className="flex flex-col space-y-4">
              <button
                onClick={() => scrollToSection("how-it-works")}
                className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium text-right py-2"
              >
                كيف تعمل الأداة
              </button>
              <button
                onClick={() => scrollToSection("analyzer")}
                className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium text-right py-2"
              >
                محلل الأخبار
              </button>
              <button
                onClick={() => scrollToSection("about")}
                className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium text-right py-2"
              >
                عن المشروع
              </button>
              <button
                onClick={() => scrollToSection("developer")}
                className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium text-right py-2"
              >
                المطور
              </button>
              <button
                onClick={() => scrollToSection("contact")}
                className="text-gray-600 hover:text-[#799EFF] transition-colors duration-200 font-medium text-right py-2"
              >
                تواصل معنا
              </button>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
