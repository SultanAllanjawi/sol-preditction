import { redirect } from "next/navigation";

export default function Home() {
  redirect("/dashboard/SOL-USD/overview");
}
