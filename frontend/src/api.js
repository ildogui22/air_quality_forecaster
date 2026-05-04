const BASE_URL = 'https://air-quality-forecaster-lrz1.onrender.com'

export async function fetchForecast(city) {
  const res = await fetch(`${BASE_URL}/forecast/${city}`)
  if (!res.ok) throw new Error(`Forecast unavailable for ${city}`)
  return res.json()
}

export async function fetchHistory(city, days = 30) {
  const res = await fetch(`${BASE_URL}/history/${city}?days=${days}`)
  if (!res.ok) throw new Error(`History unavailable for ${city}`)
  return res.json()
}
