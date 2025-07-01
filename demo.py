#!/usr/bin/env python3
"""
Demo script for the Advanced Agentic Document Intelligence System

This script demonstrates the system's capabilities with sample documents.
"""

import asyncio
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

async def run_demo():
    """Run a complete demo of the system."""
    
    console.print(Panel.fit(
        "[bold blue]🤖 Advanced Agentic Document Intelligence System Demo[/bold blue]\n\n"
        "This demo will:\n"
        "1. Initialize the system\n"
        "2. Create a sample document (if needed)\n"
        "3. Process the document\n"
        "4. Demonstrate QA capabilities\n\n"
        "[yellow]Note: Make sure you have set up your .env file with API keys![/yellow]",
        title="Demo Start",
        border_style="blue"
    ))
    
    # Check if .env file exists
    if not Path(".env").exists():
        console.print("[red]❌ .env file not found![/red]")
        console.print("Please copy .env.example to .env and configure your settings.")
        return
    
    try:
        # Import after env check
        from database.mongodb_client import mongodb_client
        from orchestrator import document_orchestrator
        from agents.qa.supervisor_agent import SupervisorAgent
        from database.models import UserQuery
        
        # Step 1: Initialize system
        console.print("\n[blue]Step 1: Initializing system...[/blue]")
        
        await mongodb_client.connect()
        console.print("✅ Database connected")
        
        # Step 2: Check for sample document or create one
        console.print("\n[blue]Step 2: Checking for sample documents...[/blue]")
        
        sample_file = await get_or_create_sample_document()
        console.print(f"✅ Sample document ready: {sample_file}")
        
        # Step 3: Process document
        console.print("\n[blue]Step 3: Processing document...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing document...", total=None)
            
            result = await document_orchestrator.process_document(str(sample_file))
            
            progress.update(task, description="✅ Processing complete!")
        
        if result['status'] == 'success':
            console.print(Panel.fit(
                f"[green]✅ Document processed successfully![/green]\n\n"
                f"Document ID: {result['document_id']}\n"
                f"Elements Extracted: {result['elements_extracted']}\n"
                f"Processing Time: {result['processing_time']:.2f}s",
                title="Processing Result",
                border_style="green"
            ))
        else:
            console.print(f"[red]❌ Processing failed: {result.get('error')}[/red]")
            return
        
        # Step 4: Demonstrate QA
        console.print("\n[blue]Step 4: Demonstrating QA capabilities...[/blue]")
        
        supervisor = SupervisorAgent()
        
        demo_questions = [
            "What is this document about?",
            "What are the main contributions of this paper?",
            "What datasets were used in the experiments?",
            "What are the key results and findings?"
        ]
        
        for question in demo_questions:
            console.print(f"\n[cyan]Q: {question}[/cyan]")
            
            try:
                user_query = UserQuery(query_text=question)
                
                with console.status("Thinking..."):
                    response = await supervisor.process(user_query)
                
                if response.response_type == "supervisor_response":
                    content = response.content
                    answer = content.get('answer', 'No answer generated')
                    console.print(f"[green]A: {answer}[/green]")
                    
                else:
                    console.print(f"[red]Error: {response.content}[/red]")
                    
            except Exception as e:
                console.print(f"[red]QA Error: {e}[/red]")
        
        console.print(Panel.fit(
            "[green]🎉 Demo completed successfully![/green]\n\n"
            "You can now:\n"
            "• Process more documents: python main.py process document <file>\n"
            "• Start interactive chat: python main.py qa chat\n"
            "• Ask single questions: python main.py qa ask \"your question\"",
            title="Demo Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("\nPlease check:")
        console.print("1. MongoDB is running")
        console.print("2. Cerebras or Groq API key is set in .env")
        console.print("3. All dependencies are installed")
        
    finally:
        await mongodb_client.disconnect()

async def get_or_create_sample_document():
    """Get the sample PDF document or create one if it doesn't exist."""
    
    # First, check for the research paper in tests/files/
    sample_pdf = Path("tests/files/1406.2661v1.pdf")
    
    if sample_pdf.exists():
        console.print(f"📄 Using existing research paper: {sample_pdf}")
        return sample_pdf
    
    # If not found, create a sample PDF
    console.print("📄 Research paper not found, creating sample PDF...")
    return await create_sample_pdf()

async def create_sample_pdf():
    """Create a sample PDF document for demo purposes."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except ImportError:
        console.print("[yellow]⚠️  reportlab not available, creating simple PDF with basic content[/yellow]")
        return await create_simple_pdf_fallback()
    
    # Create data directory
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_research_paper.pdf"
    
    if not sample_file.exists():
        # Create a comprehensive sample PDF document
        doc = SimpleDocTemplate(str(sample_file), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Learning Deep Features for Discriminative Localization", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Abstract
        abstract_title = Paragraph("Abstract", styles['Heading1'])
        story.append(abstract_title)
        
        abstract_text = """
        In this paper, we propose a novel approach for learning discriminative features
        for object localization without requiring bounding box annotations during training.
        Our method leverages deep convolutional neural networks with global average pooling
        to generate class activation maps that highlight discriminative regions.
        We evaluate our approach on standard benchmarks including MNIST and TFD datasets,
        demonstrating significant improvements in localization accuracy.
        """
        abstract_para = Paragraph(abstract_text, styles['Normal'])
        story.append(abstract_para)
        story.append(Spacer(1, 0.3*inch))
        
        # Introduction
        intro_title = Paragraph("1. Introduction", styles['Heading1'])
        story.append(intro_title)
        
        intro_text = """
        Object localization is a fundamental task in computer vision that involves
        identifying the location of objects within images. Traditional approaches
        require expensive bounding box annotations, which limits their scalability.
        In this work, we propose a weakly supervised approach that learns to localize
        objects using only image-level labels.
        """
        intro_para = Paragraph(intro_text, styles['Normal'])
        story.append(intro_para)
        story.append(Spacer(1, 0.3*inch))
        
        # Method
        method_title = Paragraph("2. Method", styles['Heading1'])
        story.append(method_title)
        
        method_text = """
        Our approach builds upon convolutional neural networks with global average pooling.
        Instead of using fully connected layers, we use class activation mapping (CAM)
        to identify discriminative regions. The network architecture consists of
        convolutional layers followed by global average pooling and a softmax classifier.
        """
        method_para = Paragraph(method_text, styles['Normal'])
        story.append(method_para)
        story.append(Spacer(1, 0.3*inch))
        
        # Experiments
        exp_title = Paragraph("3. Experiments", styles['Heading1'])
        story.append(exp_title)
        
        exp_text = """
        We evaluate our method on two datasets: MNIST for digit recognition and
        Toronto Faces Database (TFD) for facial expression recognition.
        """
        exp_para = Paragraph(exp_text, styles['Normal'])
        story.append(exp_para)
        story.append(Spacer(1, 0.2*inch))
        
        # Results Table
        table_title = Paragraph("Table 1: Experimental Results", styles['Heading2'])
        story.append(table_title)
        
        data = [
            ['Dataset', 'Method', 'Accuracy (%)', 'Localization Error'],
            ['MNIST', 'Baseline CNN', '95.2', '0.15'],
            ['MNIST', 'Our Method', '97.8', '0.08'],
            ['TFD', 'Baseline CNN', '89.3', '0.22'],
            ['TFD', 'Our Method', '92.7', '0.12']
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Conclusion
        conclusion_title = Paragraph("4. Conclusion", styles['Heading1'])
        story.append(conclusion_title)
        
        conclusion_text = """
        We presented a novel approach for discriminative localization that achieves
        state-of-the-art results on standard benchmarks. Our method demonstrates
        the effectiveness of class activation mapping for weakly supervised learning.
        Future work will explore applications to more complex datasets and scenarios.
        """
        conclusion_para = Paragraph(conclusion_text, styles['Normal'])
        story.append(conclusion_para)
        
        # Build PDF
        doc.build(story)
        console.print(f"✅ Created sample research paper PDF: {sample_file}")
    
    return sample_file

async def create_simple_pdf_fallback():
    """Create a simple PDF without reportlab dependency."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Ultimate fallback - create a text file and inform user
        console.print("[red]❌ Cannot create PDF without PyMuPDF or reportlab installed[/red]")
        console.print("Installing reportlab: pip install reportlab")
        
        # Create a simple text file as last resort
        data_dir = Path("data/samples")
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_file = data_dir / "sample_document.txt"
        
        sample_content = """Sample Research Paper

Abstract
This is a sample document for demonstrating the document intelligence system.

Introduction  
This document contains sample content including text, tables, and structured information.

Method
Our approach uses advanced techniques for content analysis.

Results
Dataset: MNIST, Accuracy: 97.8%
Dataset: TFD, Accuracy: 92.7%

Conclusion
The proposed method shows promising results on benchmark datasets.
"""
        
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        
        console.print(f"[yellow]⚠️  Created text file instead: {sample_file}[/yellow]")
        console.print("[yellow]Note: For best results, install reportlab or use a real PDF[/yellow]")
        return sample_file
    
    # Create simple PDF using PyMuPDF
    data_dir = Path("data/samples") 
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_research_paper.pdf"
    
    if not sample_file.exists():
        doc = fitz.open()  # Create new PDF
        page = doc.new_page()
        
        content = """Sample Research Paper

Abstract
This paper presents a novel approach for discriminative localization using deep learning.
We evaluate our method on MNIST and TFD datasets, achieving significant improvements.

1. Introduction
Object localization is a fundamental computer vision task. Our approach uses weakly
supervised learning to achieve state-of-the-art results without bounding box annotations.

2. Method  
We propose using class activation mapping (CAM) with global average pooling to
identify discriminative regions in images.

3. Experiments
Dataset: MNIST - Accuracy: 97.8%, Localization Error: 0.08
Dataset: TFD - Accuracy: 92.7%, Localization Error: 0.12

4. Conclusion
Our method demonstrates effectiveness for discriminative localization tasks.
Results show significant improvements over baseline approaches.
"""
        
        # Insert text
        page.insert_text((50, 50), content, fontsize=11)
        
        # Save PDF
        doc.save(str(sample_file))
        doc.close()
        
        console.print(f"✅ Created simple PDF: {sample_file}")
    
    return sample_file

if __name__ == "__main__":
    asyncio.run(run_demo())