export default function CitySelector({ cities, selected, onChange }) {
  return (
    <div className="selector">
      <label htmlFor="city-select">City</label>
      <select
        id="city-select"
        value={selected}
        onChange={e => onChange(e.target.value)}
      >
        {cities.map(c => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>
    </div>
  )
}
