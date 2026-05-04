import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'

// Avoids timezone shifts from new Date('YYYY-MM-DD') parsing as UTC
const formatDate = (str) => {
  const [y, m, d] = str.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })
}

export default function ForecastChart({ data, city }) {
  const sorted = [...data].sort((a, b) => a.horizon - b.horizon)
  const chartData = sorted.map(d => ({
    date: formatDate(d.target_date),
    predicted: Math.round(d.predicted * 10) / 10,
    actual: d.actual != null ? Math.round(d.actual * 10) / 10 : null,
  }))

  const forecastDate = data[0]?.forecast_date

  return (
    <div className="card">
      <h2>7-Day PM10 Forecast — {city}</h2>
      {forecastDate && (
        <p className="subtitle">As of {formatDate(forecastDate)}</p>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis unit=" μg/m³" tick={{ fontSize: 12 }} width={85} />
          <Tooltip formatter={v => [`${v} μg/m³`]} />
          <Legend />
          <ReferenceLine
            y={15} stroke="#10b981" strokeDasharray="4 4"
            label={{ value: 'WHO annual (15)', fontSize: 11, fill: '#10b981', position: 'insideTopRight' }}
          />
          <ReferenceLine
            y={45} stroke="#f59e0b" strokeDasharray="4 4"
            label={{ value: 'WHO daily (45)', fontSize: 11, fill: '#f59e0b', position: 'insideTopRight' }}
          />
          <Line
            type="monotone" dataKey="predicted" stroke="#3b82f6"
            strokeWidth={2} dot={{ r: 4 }} name="Predicted"
          />
          <Line
            type="monotone" dataKey="actual" stroke="#10b981"
            strokeWidth={2} dot={{ r: 5 }} connectNulls={false} name="Actual"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
