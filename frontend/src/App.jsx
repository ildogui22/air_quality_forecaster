import { useState, useEffect } from 'react'
import CitySelector from './components/CitySelector'
import ForecastChart from './components/ForecastChart'
import HistoryChart from './components/HistoryChart'
import { fetchForecast, fetchHistory } from './api'
import './App.css'

const CITIES = ['Berlin', 'London', 'Paris', 'Amsterdam']

export default function App() {
  const [city, setCity] = useState('Berlin')
  const [forecast, setForecast] = useState(null)
  const [history, setHistory] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setForecast(null)
    setHistory(null)

    Promise.all([
      fetchForecast(city.toLowerCase()),
      fetchHistory(city.toLowerCase(), 30),
    ])
      .then(([f, h]) => {
        setForecast(f)
        setHistory(h)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [city])

  return (
    <div className="app">
      <header className="header">
        <h1>Air Quality Forecaster</h1>
        <p>Daily PM10 predictions for European cities</p>
      </header>

      <main className="main">
        <CitySelector cities={CITIES} selected={city} onChange={setCity} />

        {loading && (
          <p className="status">
            Loading… Render may take up to 30s to wake up on the first request.
          </p>
        )}
        {error && <p className="status error">Error: {error}</p>}

        {forecast && <ForecastChart data={forecast} city={city} />}
        {history && <HistoryChart data={history} city={city} />}
      </main>

      <footer className="footer">
        <p>
          Data:{' '}
          <a href="https://waqi.info" target="_blank" rel="noreferrer">WAQI</a>
          {' · '}
          <a href="https://open-meteo.com" target="_blank" rel="noreferrer">Open-Meteo</a>
          {' · Models tracked on '}
          <a href="https://dagshub.com" target="_blank" rel="noreferrer">DagsHub</a>
        </p>
      </footer>
    </div>
  )
}
