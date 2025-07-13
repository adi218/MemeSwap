# MemeSwap 🎭

A web application that allows users to search for GIFs and swap faces (their own or friends') onto them to create personalized memes.

## 🚀 Features

- **GIF Search**: Search for GIFs using Tenor API
- **Face Detection**: Automatically detect faces in GIFs and photos
- **Face Swap**: Seamlessly swap faces while maintaining animation
- **User-Friendly**: Simple drag-and-drop interface
- **Download**: Save your creations

## 🛠️ Tech Stack

### Frontend
- **Next.js** - React framework for the web
- **Tailwind CSS** - Utility-first CSS framework
- **Shadcn/ui** - Beautiful UI components
- **Framer Motion** - Smooth animations

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.11+** - Programming language
- **MediaPipe** - Face detection and landmarks
- **OpenCV** - Image processing
- **SQLite/PostgreSQL** - Database

### AI/ML
- **MediaPipe Face Detection** - Detect faces in images
- **MediaPipe Face Mesh** - Extract facial landmarks
- **Custom Face Swap Algorithm** - Seamless face swapping

## 📁 Project Structure

```
MemeSwap/
├── frontend/                    # Next.js frontend
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Next.js pages
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Helper functions
│   │   ├── styles/            # CSS and Tailwind
│   │   └── lib/               # Third-party integrations
│   ├── public/                # Static assets
│   └── package.json
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── api/               # API routes
│   │   ├── core/              # Config, security
│   │   ├── models/            # Database models
│   │   ├── services/          # Business logic
│   │   ├── utils/             # Helper functions
│   │   └── ml/                # AI/ML components
│   ├── uploads/               # File storage
│   ├── requirements.txt
│   └── main.py
├── shared/                     # Shared types/utilities
│   └── types/
└── docs/                      # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MemeSwap
   ```

2. **Set up the frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Set up the backend**
   ```bash
   cd backend
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

4. **Set up environment variables**
   - Create `.env` files in both frontend and backend directories
   - Add your Tenor API key and other configuration

### Development

- Frontend runs on: http://localhost:3000
- Backend runs on: http://localhost:8000
- API documentation: http://localhost:8000/docs

## 📚 Learning Path

This project is designed to help you learn frontend development while building a real application. The learning path includes:

1. **HTML & CSS Basics** (Week 1)
2. **JavaScript Fundamentals** (Week 2)
3. **React Basics** (Week 3-4)
4. **Advanced React** (Week 5-6)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Tenor API](https://tenor.com/developer) for GIF search
- [MediaPipe](https://mediapipe.dev/) for face detection
- [Next.js](https://nextjs.org/) for the frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework 