
const ideaForm = document.getElementById('ideaForm');
const ideaTextarea = document.getElementById('idea');
const scoreOutputDiv = document.getElementById('scoreOutput');
const getScoreButton = document.getElementById('getScoreButton'); 


ideaForm.addEventListener('submit', async (event) => {
    event.preventDefault(); 

    const idea = ideaTextarea.value.trim(); 

    if (!idea) {
        scoreOutputDiv.innerHTML = '<p style="color: red;">Please enter an idea!</p>';
        return;
    }

    
    scoreOutputDiv.innerHTML = '<p>Predicting score... please wait.</p>';
    getScoreButton.disabled = true;

    try {
        
        const response = await fetch('http://127.0.0.1:8000/predict_feasibility/', {
            method: 'POST', 
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ idea_text: idea }) 
        });

      
        if (response.ok) {
            const data = await response.json(); 

           
            scoreOutputDiv.innerHTML = `
                <p><strong>Predicted Feasibility Score:</strong> ${data.predicted_feasibility_score}</p>
                <p><strong>Suggestion:</strong> ${data.suggestion}</p>
            `;
        } else {
           
            const errorData = await response.json();
            scoreOutputDiv.innerHTML = `<p style="color: red;">Error: ${errorData.detail || 'Could not get prediction'}</p>`;
        }
    } catch (error) {
        
        console.error('Network or API error:', error);
        scoreOutputDiv.innerHTML = `<p style="color: red;">Error: Could not connect to the AI service. Is the backend server running?</p>`;
    } finally {
        getScoreButton.disabled = false; 
    }
});const ideas = {
    social: [
      "Launch a community garden project",
      "Platform for elderly care volunteers",
      "Build a neighborhood skill exchange app"
    ],
    Tech: [
      "Develop an AI chatbot for student help",
      "Create a personal finance tracker",
      "Build a smart home automation system"
    ],
    Health: [
      "Create a mental wellness journaling app",
      "Develop a fitness challenge tracker",
      "Build an online therapy scheduler"
    ],
    Education: [
      "Create a gamified learning app",
      "Build a student project idea bank",
      "Develop a peer tutoring platform"
    ],
    Business: [
      "Launch a virtual business card platform",
      "Build a local seller marketplace",
      "Develop a startup mentor-matching site"
    ],
    other: [
      "Create an eco-friendly travel guide",
      "Design a pet adoption tracker",
      "Start a hobby exchange club for people"
    ]
  };

  const radios = document.querySelectorAll('input[name="type"]');
  const suggestionBox = document.getElementById("suggestionBox");
  const ideaInput = document.getElementById("idea");

  radios.forEach(radio => {
    radio.addEventListener("change", () => {
      const selected = radio.value;
      const selectedIdeas = ideas[selected] || [];

      
      suggestionBox.innerHTML = "";

      
      selectedIdeas.forEach(idea => {
        const btn = document.createElement("button");
        btn.textContent = idea;
        btn.onclick = () => {
          ideaInput.value = idea;
        };
        suggestionBox.appendChild(btn);
      });
    });
  });