import React from "react";

const page = async ({ params }: { params: { id: string } }) => {
  const { id } = await params;

  return <h1>User Page {id}</h1>;
};

export default page;
