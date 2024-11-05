// Define localization settings
Survey.surveyLocalization.locales["custom"] = {
    "pagePrevText": "Πίσω",
    "pageNextText": "Συνέχεια",
    "completeText": "Ολοκλήρωση",
    "prevText": "Προηγούμενο",
    "nextText": "Επόμενο",
    "progressText": "Πρόοδος",
    "completeText": "Υποβολή",
    "editText": "Επεξεργασία",
    "startSurveyText": "Ξεκινήστε",
    "surveyComplete": "Η έρευνα ολοκληρώθηκε! Ευχαριστώ πολύ για το χρόνο σας!"
};

// Set the localization to use
Survey.surveyLocalization.currentLocale = "custom";

// Function for getting a random integer
function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Function for shuffling an array (optional, but not used for answers anymore)
function shuffleArray(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

// Function for sending post request to server with survey data
function sendDataToServer(sender, options) {
    options.showDataSaving();

    // Format responses as requested
    let formattedAnswers = {};
    Object.keys(sender.data).forEach((key) => {
        if (key.startsWith('factuality-') || key.startsWith('completeness-') || key.startsWith('usefulness-')) {
            let value = sender.data[key];
            let [aspect, rest] = key.split('-');
            let [queryIndex, answerIndex] = rest.split('_');
            
            // Convert indices to integers to avoid issues with string concatenation
            queryIndex = parseInt(queryIndex);
            answerIndex = parseInt(answerIndex);

            // Simplify the formatted key using the answer index as it appears
            formattedAnswers[`${aspect}_${queryIndex}_${answerIndex}`] = value;
        }
    });

    // Collect demographic data
    let demographicData = {
        age: sender.data['age'],
        gender: sender.data['gender'],
        ai_familiarity: sender.data['ai_familiarity']
    };

    // Collect feedback data
    let feedbackComment = sender.data['feedback'] || ""; // Capture feedback comment if available

    // Prepare answers with text
    let answersWithText = test_queries.map((query, queryIndex) => ({
        query: query.query,
        answers: query.answers.map((answer, answerIndex) => ({
            answerIndex: answerIndex,
            answerText: answer
        }))
    }));

    // Prepare data for sending to server
    let dataJSON = {
        answers: formattedAnswers,
        queries: test_queries.map(query => ({
            query: query.query,
            numberOfAnswers: query.answers.length
        })),
        answersWithText: answersWithText,  // Add answers with text here
        survey: test_queries.length, // Total number of queries
        testN: t_idx,
        demographics: demographicData, // Add demographic data here
        feedback: feedbackComment // Include feedback comment here
    };

    $.ajax({
        type: "POST",
        url: "/postmethod",
        contentType: "application/json",
        data: JSON.stringify(dataJSON),
        dataType: "json",
        success: function(response) {
            console.log(response);
            options.showDataSavingSuccess();
        },
        error: function(err) {
            console.log(err);
            options.showDataSavingError();
        }
    });
}

// Style your survey
Survey.StylesManager.applyTheme();

// Define the main structure of the survey
var defsurveyJSON = {
    title: "Έρευνα Αξιολόγησης Απαντήσεων σε Ερωτήσεις",
    description: "Αξιολογείστε εναλλακτικές απαντήσεις σε μία σείρα από ερωτήσεις.",
    pages: [
        {
            name: "Introduction",
            elements: [
                {
                    type: "html",
                    name: "Info",
                    html: '<p>Αυτή η σελίδα έχει δημιουργηθεί ως μέρος της διπλωματικής μου εργασίας.</p><p>Καλείστε να αξιολογήσετε εναλλακτικές απαντήσεις της <strong>Θεανώς, ενός ψηφιακού βοηθού για τον COVID-19 που αναπτύχθηκε κατά τη διάρκεια της πανδημίας</strong>, σε 6 διαφορετικές ερωτήσεις χρηστών. Για κάθε απάντηση, θα πρέπει να αξιολογήσετε τα εξής χαρακτηριστικά σε μια κλίμακα από το 1 έως το 5: </li><li><strong>Πληρότητα:</strong> Σε ποιο βαθμό η απάντηση παρέχει όλες τις απαραίτητες πληροφορίες για την αποσαφήνιση της ερώτησης; </li><li><strong>Ακρίβεια Πληροφοριών:</strong>Σε ποιο βαθμό θεωρείτε ότι οι πληροφορίες της απάντησης είναι αληθείς και ανταποκρίνονται στην πραγματικότητα; </li><li><strong>Συνολική χρησιμότητα:</strong> Πόσο χρήσιμη θεωρείτε ότι είναι συνολικά η απάντηση;</li></ul></p><p><strong>(!) Σημαντικές Διευκρινίσεις (!)</strong>: <ol><li>Κατά την αξιολόγηση, θα πρέπει να λάβετε υπόψη το <strong>χρονικό πλαίσιο της περιόδου της πανδημίας (2020-2023)</strong>; διαφορετικά, πολλές ερωτήσεις αλλά και απαντήσεις ίσως φανούν παρωχημένες ή μη αληθείς!.</li><li>Μπορείτε να χρησιμοποιήσετε μια <strong>μηχανή αναζήτησης</strong> για να ελέγξετε τους ισχυρισμούς μιας απάντησης που δεν είναι προφανείς για εσάς.</li><li>Ορισμένες ερωτήσεις χρηστών ενδέχεται να περιέχουν <strong>γραμματικά, συντακτικά ή ορθογραφικά λάθη</strong>. Εάν η ερώτηση είναι διφορούμενη αλλά έχει μερικές λογικές ερμηνείες, επιμείνετε στην ερμηνεία που θεωρείτε πιο πιθανή. </li><li>Οι απαντήσεις μπορεί να φαίνονται <strong>παρόμοιες μεταξύ τους</strong>, οπότε δώστε ιδιαίτερη προσοχή κατά την αξιολόγηση.</li></ol></p><br><br><br><p><strong>Υποστηριζόμενοι Browsers:</strong> Firefox, Chrome, Safari</p><p><strong>Εκτιμώμενος χρόνος ολοκλήρωσης της έρευνας:</strong> 10-15 λεπτά</p><p><strong>Για τυχόν απορίες επικοινωνήστε εδώ:</strong> <a href="mailto:pangriziotis@gmail.com">pangriziotis@gmail.com</a></p>'
                }
            ],
            title: "Εισαγωγή"
        },
        {
            name: "Demographics",
            elements: [
                {
                    type: "radiogroup",
                    name: "age",
                    title: "Πόσο χρονών είστε;",
                    isRequired: true,
                    choices: [
                        { value: "under_20", text: "Κάτω από 20" },
                        { value: "20_30", text: "20-30" },
                        { value: "30_40", text: "30-40" },
                        { value: "40_plus", text: "40+" }
                    ]
                },
                {
                    type: "radiogroup",
                    name: "gender",
                    title: "Ποιο είναι το φύλο σας;",
                    isRequired: true,
                    choices: [
                        { value: "male", text: "Άντρας" },
                        { value: "female", text: "Γυναίκα" },
                        { value: "other", text: "Άλλο" }
                    ]
                },
                {
                    type: "rating",
                    name: "ai_familiarity",
                    title: "Πόσο εξοικειωμένοι είστε με τη μηχανική μάθηση και την τεχνητή νοημοσύνη;",
                    isRequired: true,
                    minRateDescription: "Καθόλου",
                    maxRateDescription: "Απόλυτα"
                }
            ],
            title: "Δημογραφικά Στοιχεία",
            description: "Παρακαλώ απαντήστε στις παρακάτω δημογραφικές ερωτήσεις."
        },
        {
            name: "Feedback",
            elements: [
                {
                    type: "comment",
                    name: "feedback",
                    title: "Εδώ μπορείτε να αφήσετε οποιοδήποτε επιπλέον σχόλιο για την έρευνα (προαιρετικό).",
                    hideNumber: true
                }
            ],
            title: "Σχόλια",
            description: ""
        }
    ],
    showQuestionNumbers: "onPage",
    showProgressBar: "bottom",
    firstPageIsStarted: true
};

// Define the format of the page that includes each text with the rating questions
var defTestPage = {
    name: "Test ",
    questions: [
        {
            type: "html",
            name: "text-",
            html: "test"
        }
        // Rating questions will be added dynamically
    ],
    title: "Αξιολόγηση ",
    description: "Αξιολογήστε τις ακόλουθες απαντήσεις με βάση τα κριτήρια που σας δίνονται."
};

// List of test queries to choose from randomly
var test_queries = [];
var t_idx = -1;

// Load the test data and initialize the survey
$.when(
    $.getJSON('static/js/test_out_of_scope_threashold.json'),
    $.getJSON('static/js/test_faq_threashold.json')
).done(function(out_of_scope_data, general_faq_data) {
    // Extract the data from the responses
    out_of_scope_data = out_of_scope_data[0];
    general_faq_data = general_faq_data[0];

    // Randomly shuffle the entire datasets (if desired, but not for answers anymore)
    let shuffledOutOfScope = shuffleArray(out_of_scope_data.slice());
    let shuffledGeneralFaq = shuffleArray(general_faq_data.slice());

    // Select 3 random queries from each category
    let selectedOutOfScopeQueries = shuffledOutOfScope.slice(0, 3);
    let selectedGeneralFaqQueries = shuffledGeneralFaq.slice(0, 3);

    // Combine and shuffle the selected queries
    let queries = selectedOutOfScopeQueries.concat(selectedGeneralFaqQueries);
    queries = shuffleArray(queries); // Shuffle combined queries

    // Set test_queries for sending in JSON
    test_queries = queries; // Ensure test_queries is defined and populated

    let surveyJSON = JSON.parse(JSON.stringify(defsurveyJSON));

    // Add a test page for each query
    queries.forEach((query, i) => {
        let testPage = JSON.parse(JSON.stringify(defTestPage));
        testPage.name += (i + 1);
        testPage.title += (i + 1);

        // Add query and answers to the page
        let queryHtml = `<p><strong>Ερώτηση:</strong> ${query.query}</p>`;
        query.answers.forEach((answer, idx) => {
            queryHtml += `<p><strong>Απάντηση ${idx + 1}:</strong> ${answer}</p>`;

            // Add rating questions for each alternative answer
            testPage.questions.push(
                {
                    type: "rating",
                    name: `completeness-${i}_${idx}`,
                    title: `Πόσο πλήρης θεωρείτε ότι είναι η Απάντηση ${idx + 1};`,
                    isRequired: true,
                    rateMin: 1,
                    rateMax: 5,
                    rateStep: 1,
                    minRateDescription: "Καθόλου",
                    maxRateDescription: "Απόλυτα"
                },
                {
                    type: "rating",
                    name: `factuality-${i}_${idx}`,
                    title: `Σε ποιο βαθμό θεωρείτε ότι οι πληροφορίες της Απάντησης ${idx + 1} είναι αληθείς και ανταποκρίνονται στην πραγματικότητα;`,
                    isRequired: true,
                    rateMin: 1,
                    rateMax: 5,
                    rateStep: 1,
                    minRateDescription: "Καθόλου",
                    maxRateDescription: "Απόλυτα"
                },
                {
                    type: "rating",
                    name: `usefulness-${i}_${idx}`,
                    title: `Πόσο χρήσιμη θεωρείτε ότι είναι συνολικά η Απάντηση ${idx + 1};`,
                    isRequired: true,
                    rateMin: 1,
                    rateMax: 5,
                    rateStep: 1,
                    minRateDescription: "Καθόλου",
                    maxRateDescription: "Απόλυτα"
                }
            );
        });

        // Set the query HTML content
        testPage.questions[0].html = queryHtml;

        // Ensure unique names for each question
        testPage.questions.forEach((q, idx) => {
            q.name += (i + 1);
        });

        surveyJSON.pages.splice(i + 2, 0, testPage);
    });

    // Initialize survey
    var survey = new Survey.Model(surveyJSON);
    $("#surveyContainer").Survey({
        model: survey,
        onComplete: sendDataToServer.bind(this)
    });
});
