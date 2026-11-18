package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// forecastStage refits the model that had the lowest TEST RMSE on ALL available
// observations (train + test) and projects it forward. Hyperparameters (the ridge
// alpha) are the ones already chosen on the training data only; nothing is tuned
// on the forecast. Other models are forecast too, purely for comparison.
//
// The "indicative band" is best-model forecast +/- that model's test RMSE. It is a
// rough yardstick of past out-of-sample error, NOT a statistical prediction interval.
func forecastStage(d *dataset, tAll []float64, results []result, best, endYear int, outDir string) error {
	n := len(d.y)
	lastYear := d.years[n-1]
	if endYear <= lastYear {
		return fmt.Errorf("-forecast-end (%d) must be after the last observed year (%d)", endYear, lastYear)
	}
	var fYears []int
	var fT []float64
	for yr := lastYear + 1; yr <= endYear; yr++ {
		fYears = append(fYears, yr)
		fT = append(fT, float64(yr-d.years[0]))
	}

	// Refit every model on all observations (same spec; ridge keeps its alpha).
	preds := make([][]float64, len(results))
	for i, r := range results {
		p, _, err := r.Fit(tAll, d.y)
		if err != nil {
			return fmt.Errorf("refit %s on all data: %w", r.Name, err)
		}
		preds[i] = p(fT)
	}
	bestPred := preds[best]
	band := results[best].Test.RMSE
	lo := make([]float64, len(fT))
	hi := make([]float64, len(fT))
	for i := range fT {
		lo[i] = math.Max(0, bestPred[i]-band)
		hi[i] = bestPred[i] + band
	}

	fmt.Printf("\n=== Forecast %d-%d ===\n", fYears[0], endYear)
	fmt.Printf("Model used: %s (lowest test RMSE = %.3f), refitted on all %d observations (%d-%d)",
		results[best].Name, results[best].Test.RMSE, n, d.years[0], lastYear)
	if !math.IsNaN(results[best].Alpha) {
		fmt.Printf(", alpha=%g (chosen earlier on training data only)", results[best].Alpha)
	}
	fmt.Println()
	fmt.Printf("Last actual value: %d = %.3f Mt CO2e\n", lastYear, d.y[n-1])
	fmt.Printf("%-6s %-12s %-12s %-12s", "Year", "Forecast", "Indic. low", "Indic. high")
	for _, r := range results {
		fmt.Printf(" %-16s", r.Name)
	}
	fmt.Println()
	for i, yr := range fYears {
		fmt.Printf("%-6d %-12.2f %-12.2f %-12.2f", yr, bestPred[i], lo[i], hi[i])
		for j := range results {
			fmt.Printf(" %-16.2f", preds[j][i])
		}
		fmt.Println()
	}

	// ---- CSV ----
	head := []string{"Year", "Type", "Horizon_Years", "Model_Used", "Forecast_Mt_CO2e",
		"Indicative_Low_Mt_CO2e", "Indicative_High_Mt_CO2e"}
	for _, r := range results {
		head = append(head, "Forecast_"+strings.NewReplacer(" ", "_", "(", "", ")", "").Replace(r.Name))
	}
	rows := [][]string{head}
	for i, yr := range fYears {
		row := []string{strconv.Itoa(yr), "Forecast", strconv.Itoa(i + 1), results[best].Name,
			fmtF(bestPred[i]), fmtF(lo[i]), fmtF(hi[i])}
		for j := range results {
			row = append(row, fmtF(preds[j][i]))
		}
		rows = append(rows, row)
	}
	csvPath := filepath.Join(outDir, "forecast_2023_2030.csv")
	if err := writeCSV(csvPath, rows); err != nil {
		return err
	}

	// ---- chart ----
	svgPath := filepath.Join(outDir, "co2_forecast.svg")
	var others []series
	palette := []string{"#c0392b", "#27ae60", "#8e44ad", "#7f8c8d"}
	for j, r := range results {
		if j == best {
			continue
		}
		others = append(others, series{r.Name, preds[j], palette[j%len(palette)]})
	}
	if err := writeSVG(svgPath, d, fYears, bestPred, lo, hi, results[best].Name, band, others); err != nil {
		return err
	}
	fmt.Printf("Saved: %s and %s\n", csvPath, svgPath)
	return nil
}

type series struct {
	Name  string
	Y     []float64
	Color string
}

func writeSVG(path string, d *dataset, fYears []int, fc, lo, hi []float64, bestName string, band float64, others []series) error {
	const (
		W, H           = 960.0, 560.0
		ml, mr, mt, mb = 70.0, 30.0, 70.0, 150.0
	)
	pw, ph := W-ml-mr, H-mt-mb
	x0, x1 := float64(d.years[0]), float64(fYears[len(fYears)-1])

	yMax := 0.0
	for _, v := range d.y {
		yMax = math.Max(yMax, v)
	}
	for _, v := range hi {
		yMax = math.Max(yMax, v)
	}
	for _, s := range others {
		for _, v := range s.Y {
			yMax = math.Max(yMax, v)
		}
	}
	yMax = math.Ceil(yMax/100) * 100

	px := func(yr float64) float64 { return ml + (yr-x0)/(x1-x0)*pw }
	py := func(v float64) float64 { return mt + ph - v/yMax*ph }

	var b strings.Builder
	f := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }
	f(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" width="%.0f" height="%.0f" font-family="Arial, Helvetica, sans-serif">`+"\n", W, H, W, H)
	f(`<rect width="100%%" height="100%%" fill="white"/>` + "\n")
	f(`<text x="%.0f" y="28" font-size="18" font-weight="bold" text-anchor="middle">Indonesia CO2 emissions: actual (%d-%d) vs forecast (%d-%d)</text>`+"\n",
		W/2, d.years[0], d.years[len(d.y)-1], fYears[0], fYears[len(fYears)-1])
	f(`<text x="%.0f" y="48" font-size="12" fill="#555" text-anchor="middle">Solid line = observed data. Dashed lines and shaded area = model projections, not observations.</text>`+"\n", W/2)

	// forecast region shading (starts after the last actual year)
	xs := px(float64(d.years[len(d.y)-1]))
	f(`<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#fff4e0"/>`+"\n", xs, mt, ml+pw-xs, ph)
	f(`<text x="%.1f" y="%.1f" font-size="12" fill="#b9770e" text-anchor="middle">FORECAST</text>`+"\n", (xs+ml+pw)/2, mt+16)

	// gridlines + y ticks
	for v := 0.0; v <= yMax+1e-9; v += 100 {
		f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e5e5e5"/>`+"\n", ml, py(v), ml+pw, py(v))
		f(`<text x="%.1f" y="%.1f" font-size="11" text-anchor="end" fill="#333">%.0f</text>`+"\n", ml-6, py(v)+4, v)
	}
	// x ticks every 10 years
	for yr := d.years[0]; float64(yr) <= x1; yr += 10 {
		f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#999"/>`+"\n", px(float64(yr)), mt+ph, px(float64(yr)), mt+ph+5)
		f(`<text x="%.1f" y="%.1f" font-size="11" text-anchor="middle" fill="#333">%d</text>`+"\n", px(float64(yr)), mt+ph+20, yr)
	}
	// axes
	f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333"/>`+"\n", ml, mt, ml, mt+ph)
	f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#333"/>`+"\n", ml, mt+ph, ml+pw, mt+ph)
	f(`<text x="%.1f" y="%.1f" font-size="12" text-anchor="middle" transform="rotate(-90 18 %.1f)">CO2 emissions (Mt CO2e)</text>`+"\n", 18.0, mt+ph/2, mt+ph/2)
	f(`<text x="%.1f" y="%.1f" font-size="12" text-anchor="middle">Year</text>`+"\n", ml+pw/2, mt+ph+40)

	// indicative band (polygon: upper left->right, lower right->left), anchored at last actual point
	lastY := float64(d.years[len(d.y)-1])
	lastV := d.y[len(d.y)-1]
	f(`<polygon fill="#f39c12" fill-opacity="0.22" stroke="none" points="%.1f,%.1f `, px(lastY), py(lastV))
	for i, yr := range fYears {
		f("%.1f,%.1f ", px(float64(yr)), py(hi[i]))
	}
	for i := len(fYears) - 1; i >= 0; i-- {
		f("%.1f,%.1f ", px(float64(fYears[i])), py(lo[i]))
	}
	f(`"/>` + "\n")

	// other models (thin dashed), connected from last actual point
	for _, s := range others {
		f(`<polyline fill="none" stroke="%s" stroke-width="1.3" stroke-dasharray="3 3" points="%.1f,%.1f `, s.Color, px(lastY), py(lastV))
		for i, yr := range fYears {
			f("%.1f,%.1f ", px(float64(yr)), py(s.Y[i]))
		}
		f(`"/>` + "\n")
	}

	// actual data: solid line + filled circles
	f(`<polyline fill="none" stroke="#1f4e79" stroke-width="2.2" points="`)
	for i, v := range d.y {
		f("%.1f,%.1f ", px(float64(d.years[i])), py(v))
	}
	f(`"/>` + "\n")
	for i, v := range d.y {
		f(`<circle cx="%.1f" cy="%.1f" r="2.6" fill="#1f4e79"/>`+"\n", px(float64(d.years[i])), py(v))
	}

	// best-model forecast: thick dashed orange line + hollow diamonds
	f(`<polyline fill="none" stroke="#e67e22" stroke-width="2.6" stroke-dasharray="7 4" points="%.1f,%.1f `, px(lastY), py(lastV))
	for i, yr := range fYears {
		f("%.1f,%.1f ", px(float64(yr)), py(fc[i]))
	}
	f(`"/>` + "\n")
	for i, yr := range fYears {
		cx, cy := px(float64(yr)), py(fc[i])
		f(`<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="white" stroke="#e67e22" stroke-width="2"/>`+"\n",
			cx, cy-5, cx+5, cy, cx, cy+5, cx-5, cy)
	}

	// legend
	ly := mt + ph + 62
	f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#1f4e79" stroke-width="2.2"/><circle cx="%.1f" cy="%.1f" r="2.6" fill="#1f4e79"/>`+"\n", ml, ly, ml+34, ly, ml+17, ly)
	f(`<text x="%.1f" y="%.1f" font-size="12">Actual data (observed)</text>`+"\n", ml+42, ly+4)
	f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#e67e22" stroke-width="2.6" stroke-dasharray="7 4"/>`+"\n", ml+230, ly, ml+264, ly)
	f(`<text x="%.1f" y="%.1f" font-size="12">Forecast: %s (best on test set)</text>`+"\n", ml+272, ly+4, bestName)
	f(`<rect x="%.1f" y="%.1f" width="34" height="12" fill="#f39c12" fill-opacity="0.22"/>`+"\n", ml+560, ly-6)
	f(`<text x="%.1f" y="%.1f" font-size="12">Indicative band: forecast +/- %.1f (test RMSE)</text>`+"\n", ml+602, ly+4, band)
	lx := ml
	ly2 := ly + 24
	for _, s := range others {
		f(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.3" stroke-dasharray="3 3"/>`+"\n", lx, ly2, lx+34, ly2, s.Color)
		f(`<text x="%.1f" y="%.1f" font-size="12">%s (comparison)</text>`+"\n", lx+42, ly2+4, s.Name)
		lx += 215
	}
	f(`<text x="%.1f" y="%.1f" font-size="11" fill="#666">The band is a rough guide from past test error, not a statistical prediction interval. Projections are not guaranteed future emissions.</text>`+"\n", ml, ly2+30)
	f("</svg>\n")

	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}
