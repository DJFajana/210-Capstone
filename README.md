# 210-Capstone


Files, version control and an auxiliary base for documents outside of the Google Drive. 

# Project Name

A modern web application deployed on AWS, leveraging FastAPI, Redis, and a React frontend. This architecture is designed for scalability, performance, and simplicity.

## Architecture Overview

- **End User**: Interacts with the web application through a browser.
- **Application Load Balancer (ALB)**: Distributes incoming traffic to backend services.
- **EC2 (or ECS with Fargate) running FastAPI**: Hosts the backend API, handling business logic and user requests.
- **Amazon ElastiCache (Redis)**: Provides in-memory caching for fast data retrieval and performance optimization.
- **Amazon RDS (PostgreSQL/MySQL)**: Stores persistent data, handling relational database management.
- **Web Application (ReactJS on S3/CloudFront or EC2)**: Serves the frontend user interface, delivering a seamless user experience.

## Deployment Steps

1. **Frontend Deployment**:
   - Host ReactJS application on S3 (static site) and distribute via CloudFront for low-latency access.
   - Alternatively, deploy on an EC2 instance for dynamic rendering.

2. **Backend API Deployment**:
   - Deploy FastAPI app on EC2 instances or use ECS with Fargate for container orchestration.
   - Attach the EC2/ECS service to the Application Load Balancer (ALB).

3. **Caching Layer**:
   - Set up Amazon ElastiCache with Redis to cache frequently accessed data and reduce database load.

4. **Database Setup**:
   - Deploy Amazon RDS with PostgreSQL or MySQL, depending on your application's needs.
   - Ensure proper security groups and parameter groups are configured.

## AWS Services Used
- **Amazon EC2/ECS**
- **Amazon ElastiCache (Redis)**
- **Amazon RDS (PostgreSQL/MySQL)**
- **AWS S3**
- **AWS CloudFront**
- **AWS Application Load Balancer (ALB)**

## Prerequisites
- AWS CLI configured with appropriate permissions.
- Docker (if deploying with ECS/Fargate).
- Node.js & React installed for building the frontend.

## Getting Started
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd project-directory
   ```

2. **Frontend:**
   - Navigate to the frontend directory and build the React app.
     ```bash
     npm install
     npm run build
     ```
   - Deploy to S3 or your chosen hosting service.

3. **Backend:**
   - Navigate to the backend directory and set up your virtual environment.
     ```bash
     pip install -r requirements.txt
     ```
   - Deploy to EC2 or ECS using Docker (if applicable).

4. **Environment Variables:**
   - Ensure your environment variables (DB credentials, Redis endpoint, etc.) are securely managed and loaded during deployment.

## Contributing
Feel free to open issues or submit pull requests. Let's make this project better together!

## License
MIT License. See `LICENSE` file for more information.

---

What do you think? Any tweaks or additions you’d like to make? Let’s get this README perfect! 😊

