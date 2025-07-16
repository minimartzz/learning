import Link from "next/link";
import Image from "next/image";
import React from "react";
import { auth, signOut, signIn } from "@/auth";

const Navbar = () => {
  const session = await auth()

  return (
    <header className="px-5 py-3 bg-white shadow-sm font-work-sans">
      <nav className="flex justify-between items-center">
        <Link href="/">
          <Image src="/logo.png" alt="logo" width={144} height={30} />
        </Link>

        <div className="flex item-center gap-5">
          {session && session?.user ? (
            <>
              // Link to create
              <Link href="/startup/create">
                <span>Create</span>
              </Link>

              // Sign out button
              <button onClick={signOut}>
                <span>Logout</span>
              </button>

              // User info
              <Link href={`/user/${session?.id}`}>
                <span>{session?.user?.name}</span> 
              </Link>
            </>
          ) : (

          )}
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
