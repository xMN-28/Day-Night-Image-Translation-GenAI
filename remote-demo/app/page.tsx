import { Demo } from "@/components/demo";

export default function Home() {
  return <Demo accessCodeRequired={Boolean(process.env.DEMO_ACCESS_CODE)} />;
}
