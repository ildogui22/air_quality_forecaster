import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'

const formatDate = (str) => {
  const [y, m, d] = str.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })
}

export default function HistoryChart({ data, city }) {
  // API returns newest-first; reverse for chronological display
  const chartData = [...data].reverse().map(d => ({
    date: formatDate(d.date),
    pm10: d.pm10 != null ? Math.round(d.pm10 * 10) / 10 : null,
  }))

  return (
    <div className="card">
      <h2>PM10 History (Last 30 Days) — {city}</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} interval="preserveStartEnd" />
          <YAxis unit=" μg/m³" tick={{ fontSize: 12 }} width={85} />
          <Tooltip formatter={v => [`${v} μg/m³`]} />
          <ReferenceLine
            y={15} stroke="#10b981" strokeDasharray="4 4"
            label={{ value: 'WHO annual (15)', fontSize: 11, fill: '#10b981', position: 'insideTopRight' }}
          />
          <ReferenceLine
            y={45} stroke="#f59e0b" strokeDasharray="4 4"
            label={{ value: 'WHO daily (45)', fontSize: 11, fill: '#f59e0b', position: 'insideTopRight' }}
          />
          <Line
            type="monotone" dataKey="pm10" stroke="#6366f1"
            strokeWidth={2} dot={false} name="PM10"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
