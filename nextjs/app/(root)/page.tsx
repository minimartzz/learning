import Image from "next/image";
import Hello from "@/app/components/hello";

export default function Home() {
  console.log("I am a Server component");

  return (
    <>
      <h1>Something here</h1>
      <Hello />
      <a href="/dashboard">To Dashboard</a>
    </>
  );
}
