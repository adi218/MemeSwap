from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import requests
from app.core.config import settings

router = APIRouter()

@router.get("/search-gifs")
async def search_gifs(
    query: str = Query(..., description="Search query for GIFs"),
    limit: int = Query(10, ge=1, le=50, description="Number of GIFs to return")
):
    """
    Search for GIFs using Tenor API.
    
    Args:
        query: Search query for GIFs
        limit: Number of GIFs to return (1-50)
    
    Returns:
        JSON response with GIF search results
    """
    if not settings.TENOR_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Tenor API key not configured"
        )
    
    try:
        # Tenor API endpoint
        url = "https://tenor.googleapis.com/v2/search"
        params = {
            "q": query,
            "key": settings.TENOR_API_KEY,
            "limit": limit,
            "media_filter": "gif"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant GIF data
        gifs = []
        for result in data.get("results", []):
            # Get the best quality GIF URL available
            media_formats = result.get("media_formats", {})
            
            # Try different quality options in order of preference
            gif_url = None
            if "gif" in media_formats:
                gif_url = media_formats["gif"].get("url")
            elif "mediumgif" in media_formats:
                gif_url = media_formats["mediumgif"].get("url")
            elif "tinygif" in media_formats:
                gif_url = media_formats["tinygif"].get("url")
            
            # Only add GIFs that have a valid URL
            if gif_url:
                gif_data = {
                    "id": result.get("id"),
                    "title": result.get("title") or f"GIF {result.get('id')}",
                    "url": gif_url,
                    "proxy_url": f"/api/gif/proxy-gif?url={gif_url}",
                    "preview_url": media_formats.get("tinygif", {}).get("url"),
                    "width": media_formats.get("gif", {}).get("dims", [0, 0])[0] or 270,
                    "height": media_formats.get("gif", {}).get("dims", [0, 0])[1] or 270
                }
                gifs.append(gif_data)
        
        return {
            "gifs": gifs,
            "total_results": len(gifs),
            "query": query
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching GIFs: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/proxy-gif")
async def proxy_gif(url: str = Query(..., description="URL of the GIF to proxy")):
    """
    Proxy a GIF through our server to avoid CORS issues.
    
    Args:
        url: The URL of the GIF to proxy
    
    Returns:
        The GIF data as a streaming response
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type=response.headers.get("content-type", "image/gif"),
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error proxying GIF: {str(e)}") 