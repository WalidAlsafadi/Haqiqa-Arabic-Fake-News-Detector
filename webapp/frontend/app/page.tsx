import Header from "@/components/Header";
import Hero from "@/components/Hero";
import NewsAnalyzer from "@/components/NewsAnalyzer";
import About from "@/components/About";
import HowItWorks from "@/components/HowItWorks";
import Developer from "@/components/Developer";
import Contact from "@/components/Contact";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      <main className="overflow-hidden">
        <Hero />
        <NewsAnalyzer />
        <HowItWorks />
        <About />
        <Developer />
        <Contact />
      </main>
      <Footer />
    </div>
  );
}
