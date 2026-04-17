type Props = {
  className?: string;
};

/**
 * Compact robot-head mark for the Run action: dome helmet, antenna, visor eyes.
 * White on emerald buttons; eyes use emerald-600 to match the button.
 */
export function AndroidRobotIcon({ className }: Props) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path stroke="currentColor" strokeWidth="1.35" strokeLinecap="round" d="M12 2.75V5" />
      <circle cx="12" cy="2.25" r="1" fill="currentColor" />
      <path
        fill="currentColor"
        d="M12 5.75c-3.65 0-6.6 2.65-6.6 5.9V17.2c0 1 .8 1.8 1.8 1.8h9.6c1 0 1.8-.8 1.8-1.8v-5.55c0-3.25-2.95-5.9-6.6-5.9z"
      />
      <circle cx="9.35" cy="12.15" r="1.35" className="fill-emerald-600" />
      <circle cx="14.65" cy="12.15" r="1.35" className="fill-emerald-600" />
      <path
        stroke="currentColor"
        strokeWidth="1.15"
        strokeLinecap="round"
        d="M9.25 16.25h5.5M10 17.9h4"
      />
      <circle cx="5.85" cy="11.5" r="0.9" fill="currentColor" />
      <circle cx="18.15" cy="11.5" r="0.9" fill="currentColor" />
    </svg>
  );
}
