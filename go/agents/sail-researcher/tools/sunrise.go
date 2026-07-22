package tools

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/nathan-osman/go-sunrise"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"googlemaps.github.io/maps"
)

// SunriseArgs defines the arguments for the get_sunrise_sunset tool.
type SunriseArgs struct {
	Latitude  float64 `json:"latitude" description:"Decimal latitude"`
	Longitude float64 `json:"longitude" description:"Decimal longitude"`
	Date      string  `json:"date" description:"Date in YYYY-MM-DD format"`
}

// SunriseResult defines the response structure for the get_sunrise_sunset tool.
type SunriseResult struct {
	Date     string `json:"date"`
	Sunrise  string `json:"sunrise"`
	Sunset   string `json:"sunset"`
	TimeZone string `json:"time_zone"`
}

// TimezoneClient defines the interface for the Google Maps Timezone API client.
type TimezoneClient interface {
	Timezone(ctx context.Context, r *maps.TimezoneRequest) (*maps.TimezoneResult, error)
}

// SunriseProvider implements the get_sunrise_sunset tool.
type SunriseProvider struct {
	client TimezoneClient
}

// Close closes the underlying client connection.
func (sp *SunriseProvider) Close() error {
	return nil
}

// NewSunriseTool creates a new ADK tool for calculating sunrise and sunset times.
func NewSunriseTool(apiKey string) (tool.Tool, *SunriseProvider, error) {
	c, err := maps.NewClient(maps.WithAPIKey(apiKey))
	if err != nil {
		return nil, nil, fmt.Errorf("creating maps client: %w", err)
	}

	sp := &SunriseProvider{
		client: c,
	}
	t, err := functiontool.New(functiontool.Config{
		Name:        "get_sunrise_sunset",
		Description: "Retrieves sunrise and sunset times for a specific location and date. Returns times in the location's local timezone.",
	}, sp.GetSunriseSunset)
	return t, sp, err
}

func (sp *SunriseProvider) GetSunriseSunset(ctx agent.Context, args SunriseArgs) (SunriseResult, error) {
	targetDate, err := time.Parse("2006-01-02", args.Date)
	if err != nil {
		return SunriseResult{}, fmt.Errorf("%w: %v", ErrInvalidDate, err)
	}

	// 1. Calculate Sunrise/Sunset (Returns UTC)
	rise, set := sunrise.SunriseSunset(
		args.Latitude,
		args.Longitude,
		targetDate.Year(),
		targetDate.Month(),
		targetDate.Day(),
	)

	// 2. Fetch Actual Timezone with Fallback
	var loc *time.Location
	var timeZoneID string

	tzReq := &maps.TimezoneRequest{
		Location:  &maps.LatLng{Lat: args.Latitude, Lng: args.Longitude},
		Timestamp: targetDate, // Use target date for correct DST
	}

	tzResult, err := sp.client.Timezone(context.Background(), tzReq)
	if err != nil {
		// Fallback to LMT (Local Mean Time) approximation
		// Offset = Round(Longitude / 15)
		offsetHours := int(math.Round(args.Longitude / 15.0))
		loc = time.FixedZone("LMT", offsetHours*3600)
		timeZoneID = "LMT (Approximate)"
	} else {
		// Use API result
		totalOffsetSeconds := tzResult.DstOffset + tzResult.RawOffset
		loc = time.FixedZone(tzResult.TimeZoneName, totalOffsetSeconds)
		timeZoneID = tzResult.TimeZoneID
	}

	localRise := rise.In(loc)
	localSet := set.In(loc)

	timeFmt := "2006-01-02T15:04:05"

	return SunriseResult{
		Date:     args.Date,
		Sunrise:  localRise.Format(timeFmt),
		Sunset:   localSet.Format(timeFmt),
		TimeZone: timeZoneID,
	}, nil
}