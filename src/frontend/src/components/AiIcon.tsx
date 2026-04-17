import aiIcon from "../assets/images/artificial-intelligence.png";

type Props = {
  className?: string;
  /** Empty for decorative use next to visible text. */
  alt?: string;
};

export function AiIcon({ className, alt = "" }: Props) {
  return <img src={aiIcon} alt={alt} className={className} decoding="async" />;
}
