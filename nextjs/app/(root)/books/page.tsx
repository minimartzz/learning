import React from "react";
import books from "@/app/api/db";

const page = async () => {
  const response = await fetch("http://localhost:3000/api/books");
  if (!response.ok) {
    return <main> {response.status} </main>;
  }
  const data = await response.json();

  return (
    <main>
      <code>{JSON.stringify(data, null, 2)}</code>
    </main>
  );
};

export default page;
