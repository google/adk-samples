package tools

import (
	"fmt"
	"time"

	"github.com/tpryan/noaago"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	DefaultSearchRadius = 50
	MaxStationsToCheck  = 5
)

// TideArgs defines the arguments for the get_tides tool.
type TideArgs struct {
	Latitude  float64 `json:"latitude" description:"Decimal latitude"`
	Longitude float64 `json:"longitude" description:"Decimal longitude"`
	Date      string  `json:"date" description:"Date in YYYY-MM-DD format"`
}

// TideEvent represents a single high or low tide event.
type TideEvent struct {
	Time   string  `json:"time"`
	Type   string  `json:"type"`
	Height float64 `json:"height_ft"`
	Unit   string  `json:"unit"`
}

// TideResult defines the response structure for the get_tides tool.
type TideResult struct {
	StationName   string      `json:"station_name"`
	StationID     string      `json:"station_id"`
	DistanceMiles float64     `json:"distance_miles"`
	Tides         []TideEvent `json:"tides"`
}

// TideClient defines the interface for the NOAA API client.
type TideClient interface {
	FindStations(opts *noaago.StationOptions) (*noaago.StationResponse, error)
	GetTides(opts *noaago.TideOptions) (*noaago.TideResponse, error)
}

// TideProvider implements the get_tides tool using the NOAA CO-OPS API.
type TideProvider struct {
	client TideClient
}

// Close closes the underlying client connection.
func (tp *TideProvider) Close() error {
	return nil
}

// NewTideTool creates a new ADK tool for retrieving tide predictions.
func NewTideTool() (tool.Tool, *TideProvider, error) {
	client := noaago.NewClient()
	tp := &TideProvider{client: client}

	t, err := functiontool.New(functiontool.Config{
		Name:        "get_tides",
		Description: "Retrieves high and low tide predictions for a specific date from the nearest NOAA station.",
	}, tp.GetTides)
	return t, tp, err
}

func (tp *TideProvider) GetTides(ctx agent.Context, args TideArgs) (TideResult, error) {
	stations, err := tp.findNearbyStations(args.Latitude, args.Longitude)
	if err != nil {
		return TideResult{}, err
	}

	if len(stations) == 0 {
		return TideResult{}, ErrNotFound
	}

	var lastErr error
	for _, s := range stations {
		tides, err := tp.fetchPredictions(s, args.Date)
		if err == nil {
			return TideResult{
				StationName: s.Name,
				StationID:   s.ID,
				Tides:       tides,
			}, nil
		}
		lastErr = err
	}

	return TideResult{}, fmt.Errorf("getting tides from nearby stations. Last error: %v", lastErr)
}

func (tp *TideProvider) findNearbyStations(lat, lng float64) ([]noaago.Station, error) {
	for radius := DefaultSearchRadius; radius <= 10*DefaultSearchRadius; radius += DefaultSearchRadius {
		stationOpts := noaago.NewStationOptionsBuilder().
			Nearby(lat, lng, float64(radius)).
			Type(noaago.StationType("tidepredictions")).
			Build()

		stationsResp, err := tp.client.FindStations(stationOpts)
		if err != nil {
			return nil, fmt.Errorf("searching stations: %w", err)
		}

		if stationsResp.Count > 0 && len(stationsResp.Stations) > 0 {
			// Limit to checking 5 closest stations
			limit := MaxStationsToCheck
			if len(stationsResp.Stations) < limit {
				limit = len(stationsResp.Stations)
			}
			return stationsResp.Stations[:limit], nil
		}
	}

	return nil, nil
}

func (tp *TideProvider) fetchPredictions(station noaago.Station, dateStr string) ([]TideEvent, error) {
	parsedDate, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return nil, ErrInvalidDate
	}

	// Get for the date with a 48-hour buffer before and after.
	beginDate := parsedDate.Add(-48 * time.Hour)
	endDate := parsedDate.Add(48 * time.Hour)

	tideOpts := noaago.NewTideOptionsBuilder().
		StationID(station.ID).
		Product(noaago.ProductPredictions).
		Datum(noaago.DatumMLLW).
		Units(noaago.UnitsEnglish).
		Interval(noaago.IntervalHighLow).
		TimeZone(noaago.TimeZoneLSTLDT).
		DateRange(beginDate, endDate).
		Build()

	tideResp, err := tp.client.GetTides(tideOpts)
	if err != nil {
		return nil, err
	}

	var events []TideEvent
	for _, pt := range tideResp.GetData() {
		val, _ := pt.ValueFloat()
		events = append(events, TideEvent{
			Time:   pt.Time,
			Type:   pt.Type,
			Height: val,
			Unit:   "ft",
		})
	}
	return events, nil
}
