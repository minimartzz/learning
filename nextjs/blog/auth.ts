import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import { client } from "./sanity/lib/client";
import { AUTHOR_BY_GITHUB_ID } from "./lib/queries";
import { writeClient } from "./sanity/lib/write";

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [GitHub],
  callbacks: {
    async signIn({
      user: { name, email, image },
      profile: { id, login, bio },
    }) {
      const existingUser = await client
        .withConfig({ useCdn: false })
        .fetch(AUTHOR_BY_GITHUB_ID, {
          id,
        });

      if (!existingUser) {
        await writeClient.create({
          _type: "author",
          id,
          name,
          username: login,
          email,
          image,
          bio: bio || "",
        });
      }

      return true;
    },
    async jwt({ token, account, profile }) {
      if (account && profile) {
        const user = await client
          .withConfig({ useCdn: false })
          .fetch(AUTHOR_BY_GITHUB_ID, {
            id: profile?.id, // profile.id should already be defined if account is defined
          });

        if (user?._id) {
          token.id = user._id;
        } else {
          console.log(
            "WARNING: user._id was undefined or user was null in JWT callback!"
          );
        }
      } else {
        console.log(
          "Account or profile not available (might be token refresh). Current token.id:",
          token.id
        );
      }
      return token;
    },
    async session({ session, token }) {
      if (token?.id) {
        const newSession = { ...session, id: token.id as string };

        return newSession;
      } else {
        return session;
      }
    },
  },
});
