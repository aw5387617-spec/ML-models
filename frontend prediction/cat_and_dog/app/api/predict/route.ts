import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        // Parse the form data from the request
        const formData = await request.formData();
        const image = formData.get('image') as File | null;

        if (!image) {
            return NextResponse.json(
                { error: 'No image provided' },
                { status: 400 }
            );
        }

        // Validate file type
        if (!image.type.match('image/jpeg') && !image.type.match('image/png')) {
            return NextResponse.json(
                { error: 'Invalid file type. Please upload a JPEG or PNG image.' },
                { status: 400 }
            );
        }

        // Validate file size (max 5MB)
        if (image.size > 5 * 1024 * 1024) {
            return NextResponse.json(
                { error: 'File size exceeds 5MB limit' },
                { status: 400 }
            );
        }

        // Create a new FormData to forward the image to the Flask API
        const flaskFormData = new FormData();
        flaskFormData.append('image', image);

        // Call the Flask API for actual prediction
        const flaskResponse = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: flaskFormData,
        });

        if (!flaskResponse.ok) {
            const errorData = await flaskResponse.json().catch(() => ({}));
            return NextResponse.json(
                { error: errorData.error || `Flask API error: ${flaskResponse.status}` },
                { status: flaskResponse.status || 500 }
            );
        }

        const flaskData = await flaskResponse.json();

        if (!flaskData.success) {
            return NextResponse.json(
                { error: flaskData.error || 'Prediction failed' },
                { status: 500 }
            );
        }

        // Process the response from the Flask API
        const prediction = flaskData.prediction;
        const confidencePercent = Math.round(prediction.confidence * 100); // Convert to percentage

        // Return the prediction result in the format expected by the frontend
        return NextResponse.json({
            success: true,
            prediction: {
                animal: prediction.predicted_class.charAt(0).toUpperCase() + prediction.predicted_class.slice(1), // Capitalize first letter
                confidence: confidencePercent,
                fileName: image.name,
                fileSize: image.size,
                probabilities: prediction.probabilities
            }
        });
    } catch (error) {
        console.error('Prediction error:', error);
        return NextResponse.json(
            { error: 'Prediction failed. Please try again.' },
            { status: 500 }
        );
    }
}