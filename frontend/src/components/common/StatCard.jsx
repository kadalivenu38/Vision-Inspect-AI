import './StatCard.css';

export default function StatCard({ icon, label, value, trend, trendDirection = 'neutral', variant = 'accent' }) {
  return (
    <article className="stat-card">
      <div className={`stat-card__icon stat-card__icon--${variant}`}>
        {icon}
      </div>
      <div className="stat-card__content">
        <p className="stat-card__label">{label}</p>
        <p className="stat-card__value">{value}</p>
        {trend && (
          <p className={`stat-card__trend stat-card__trend--${trendDirection}`}>
            {trendDirection === 'up' && (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M6 2.5L10 7.5H2L6 2.5Z" fill="currentColor" />
              </svg>
            )}
            {trendDirection === 'down' && (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M6 9.5L2 4.5H10L6 9.5Z" fill="currentColor" />
              </svg>
            )}
            {trend}
          </p>
        )}
      </div>
    </article>
  );
}
