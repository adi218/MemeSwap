'use client';

import { useState } from 'react';

export default function GifSearch({ onGifSelect }) {
  const [query, setQuery] = useState('');
  const [gifs, setGifs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [imageErrors, setImageErrors] = useState({});

  const searchGifs = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setHasSearched(true);
    setImageErrors({});

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/gif/search-gifs?query=${encodeURIComponent(query)}&limit=10`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch GIFs');
      }

      const data = await response.json();
      setGifs(data.gifs);
    } catch (err) {
      setError('Error searching for GIFs: ' + err.message);
      setGifs([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    searchGifs();
  };

  const handleImageError = (gifId) => {
    setImageErrors(prev => ({ ...prev, [gifId]: true }));
  };

  const handleImageLoad = (gifId) => {
    setImageErrors(prev => ({ ...prev, [gifId]: false }));
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Search Form */}
      <form onSubmit={handleSubmit} className="mb-8">
        <div className="flex gap-4 max-w-lg mx-auto">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for GIFs (e.g., 'dancing', 'laughing', 'surprised')..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-800 placeholder-gray-500 text-lg"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-8 py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-lg"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-800 rounded-lg max-w-lg mx-auto">
          {error}
        </div>
      )}

      {/* Results */}
      {gifs.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-6 text-gray-800 text-center">
            Found {gifs.length} GIFs for "<span className="text-indigo-600">{query}</span>"
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {gifs.map((gif) => (
              <div
                key={gif.id}
                className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg hover:border-indigo-300 transition-all duration-200 bg-white cursor-pointer group"
                onClick={() => onGifSelect(gif)}
              >
                {gif.url && (
                  <div className="relative">
                    {imageErrors[gif.id] ? (
                      <div className="w-full h-48 bg-gray-100 flex items-center justify-center">
                        <div className="text-center text-gray-500">
                          <div className="text-4xl mb-2">🖼️</div>
                          <p className="text-sm">Image failed to load</p>
                        </div>
                      </div>
                    ) : (
                      <div className="relative">
                        <img
                          src={gif.proxy_url ? `http://127.0.0.1:8000${gif.proxy_url}` : gif.url}
                          alt={gif.title || 'GIF'}
                          className="w-full h-auto max-h-48 object-contain group-hover:scale-105 transition-transform duration-200"
                          style={{
                            maxWidth: '100%',
                            height: 'auto'
                          }}
                          onError={() => handleImageError(gif.id)}
                          onLoad={() => handleImageLoad(gif.id)}
                          loading="lazy"
                        />
                        {gif.width && gif.height && (
                          <div className="absolute bottom-2 right-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                            {gif.width} × {gif.height}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                <div className="p-4">
                  <h4 className="font-medium text-gray-900 truncate">
                    {gif.title || 'Untitled GIF'}
                  </h4>
                  <p className="text-sm text-gray-600">
                    {gif.width} × {gif.height}
                  </p>
                  <p className="text-xs text-indigo-600 mt-2 font-medium">
                    Click to select this GIF
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Results - Only show after search has been completed */}
      {!loading && hasSearched && gifs.length === 0 && query && !error && (
        <div className="text-center text-gray-600 py-8">
          <p className="text-lg">No GIFs found for "<span className="text-indigo-600">{query}</span>"</p>
          <p className="text-sm mt-2">Try a different search term</p>
        </div>
      )}

      {/* Initial State */}
      {!hasSearched && (
        <div className="text-center text-gray-500 py-8">
          <p className="text-lg">Enter a search term above to find GIFs</p>
          <p className="text-sm mt-2">Popular searches: dancing, laughing, surprised, shocked</p>
        </div>
      )}
    </div>
  );
} 