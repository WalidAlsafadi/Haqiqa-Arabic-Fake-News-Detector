import React from "react";
import { Cairo } from "next/font/google";
import "./globals.css";

const cairo = Cairo({
  subsets: ["arabic", "latin"],
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

export const metadata = {
  title: "حقيقة | كاشف الأخبار المزيفة",
  description:
    "حقيقة - أداة ذكية لكشف الأخبار المزيفة باللغة العربية مدعومة بتقنية AraBERT والذكاء الاصطناعي",
  keywords:
    "حقيقة, أخبار مزيفة, ذكاء اصطناعي, AraBERT, كشف الأخبار الكاذبة, فلسطين",
  authors: [{ name: "Walid Alsafadi" }],
  openGraph: {
    title: "حقيقة | كاشف الأخبار المزيفة",
    description: "أداة متقدمة لكشف الأخبار المزيفة باللغة العربية",
    type: "website",
  },
  icons: {
    icon: "/favicon.svg",
    apple: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ar" dir="rtl">
      <body className={cairo.className}>{children}</body>
    </html>
  );
}
