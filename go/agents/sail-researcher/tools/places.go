package tools

import (
	"context"
	"fmt"
	"strings"
	"time"

	places "cloud.google.com/go/maps/places/apiv1"
	"cloud.google.com/go/maps/places/apiv1/placespb"
	"github.com/googleapis/gax-go/v2"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/api/option"
	"google.golang.org/genproto/googleapis/type/latlng"
	"google.golang.org/grpc/metadata"
)

// PlacesArgs defines the arguments for the find_places_nearby tool.
type PlacesArgs struct {
	Query     string  `json:"query" description:"Text query (e.g. 'restaurants', 'marinas')."`
	Latitude  float64 `json:"latitude" description:"Latitude for location bias."`
	Longitude float64 `json:"longitude" description:"Longitude for location bias."`
	Radius    float64 `json:"radius" description:"Search radius in meters. Default 5000."`
	OpenNow   bool    `json:"open_now" description:"If true, only return places currently open."`
	MinRating float64 `json:"min_rating" description:"Minimum rating (1.0 - 5.0)."`
}

// PlaceResult represents a single place found by the search.
type PlaceResult struct {
	Name            string   `json:"name"`
	Address         string   `json:"address"`
	Latitude        float64  `json:"latitude"`
	Longitude       float64  `json:"longitude"`
	Rating          float64  `json:"rating"`
	UserRatingCount int32    `json:"user_rating_count"`
	BusinessStatus  string   `json:"business_status"`
	Types           []string `json:"types"`
	WebsiteURI      string   `json:"website_uri,omitempty"`
}

// PlacesResponse defines the response structure for the find_places_nearby tool.
type PlacesResponse struct {
	Places          []PlaceResult `json:"places"`
	DebugDurationMS int64         `json:"debug_duration_ms"`
}

// PlacesClient defines the interface for the Google Places API client.
type PlacesClient interface {
	SearchText(ctx context.Context, req *placespb.SearchTextRequest, opts ...gax.CallOption) (*placespb.SearchTextResponse, error)
	Close() error
}

// PlacesProvider implements the find_places_nearby tool using the Google Maps Places API.
type PlacesProvider struct {
	client PlacesClient
}

// Close closes the underlying client connection.
func (p *PlacesProvider) Close() error {
	return p.client.Close()
}

// NewPlacesTool creates a new ADK tool for searching places nearby.
func NewPlacesTool(ctx context.Context, apiKey string) (tool.Tool, *PlacesProvider, error) {
	var clientOpts []option.ClientOption
	if apiKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(apiKey))
	}

	c, err := places.NewClient(ctx, clientOpts...)
	if err != nil {
		return nil, nil, fmt.Errorf("creating Places client: %w", err)
	}

	p := &PlacesProvider{client: c}

	t, err := functiontool.New(functiontool.Config{
		Name:        "find_places_nearby",
		Description: "Finds places (e.g. marinas, restaurants) near a location using Google Maps Text Search. Returns specific locations with Lat/Lng.",
	}, p.FindPlaces)

	return t, p, err
}

func (p *PlacesProvider) FindPlaces(ctx agent.Context, args PlacesArgs) (PlacesResponse, error) {
	start := time.Now()

	// Default radius if 0
	radius := args.Radius
	if radius <= 0 {
		radius = 5000 // 5km default
	}

	centerPoint := &latlng.LatLng{
		Latitude:  args.Latitude,
		Longitude: args.Longitude,
	}

	circleArea := &placespb.Circle{
		Center: centerPoint,
		Radius: radius,
	}

	locationBias := &placespb.SearchTextRequest_LocationBias{
		Type: &placespb.SearchTextRequest_LocationBias_Circle{
			Circle: circleArea,
		},
	}

	// Define fields to return (FieldMask)
	// Basic fields + location + rating + website
	fieldsToRequest := []string{
		"places.displayName",
		"places.formattedAddress",
		"places.location",
		"places.rating",
		"places.userRatingCount",
		"places.businessStatus",
		"places.types",
		"places.websiteUri",
	}
	fieldMaskHeader := strings.Join(fieldsToRequest, ",")

	req := &placespb.SearchTextRequest{
		TextQuery:      args.Query,
		LocationBias:   locationBias,
		OpenNow:        args.OpenNow,
		MinRating:      args.MinRating,
		MaxResultCount: 5,
	}

	// Append FieldMask to context
	callCtx := metadata.AppendToOutgoingContext(ctx, "x-goog-fieldmask", fieldMaskHeader)

	resp, err := p.client.SearchText(callCtx, req)
	if err != nil {
		return PlacesResponse{}, fmt.Errorf("SearchText API: %w", err)
	}

	placesList := resp.Places

	results := make([]PlaceResult, 0)
	for _, pt := range placesList {
		lat := 0.0
		lng := 0.0
		if pt.Location != nil {
			lat = pt.Location.Latitude
			lng = pt.Location.Longitude
		}

		name := ""
		if pt.DisplayName != nil {
			name = pt.DisplayName.Text
		}

		results = append(results, PlaceResult{
			Name:            name,
			Address:         pt.FormattedAddress,
			Latitude:        lat,
			Longitude:       lng,
			Rating:          float64(pt.Rating),
			UserRatingCount: pt.GetUserRatingCount(),
			BusinessStatus:  pt.BusinessStatus.String(),
			Types:           pt.Types,
			WebsiteURI:      pt.WebsiteUri,
		})
	}

	return PlacesResponse{
		Places:          results,
		DebugDurationMS: time.Since(start).Milliseconds(),
	}, nil
}
