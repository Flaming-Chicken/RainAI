# RainAI Privacy Policy & Contribution Terms

**Effective Date:** 2026-10-04

RainAI is an open-source, neural-parametric soundscape synthesis project. We are committed to processing only the minimal amount of data necessary to protect our infrastructure and ensure the integrity of the community dataset.

This Privacy Policy explains what data we collect, why we collect it, and how it is used when you visit `rainai.app` or contribute audio.

## 1. Information We Process Automatically (Infrastructure Protection)

To keep the platform online and protect against abuse, denial-of-service (DDoS) attacks, and spam, our edge infrastructure automatically processes technical data:

* **Cloudflare Turnstile (Anti-Bot):** We use Cloudflare Turnstile to verify that submissions are made by humans. Turnstile performs non-interactive browser checks and processes your IP address and browser telemetry strictly for bot-mitigation. We do not use this data for tracking or advertising. (For more details, see [Cloudflare's Privacy Policy](https://www.cloudflare.com/privacypolicy/)).
* **IP Rate Limiting:** We temporarily hash and monitor incoming IP addresses to enforce rate limits (e.g., limiting the number of uploads per hour). Raw IP addresses are not stored permanently in our databases; they are used ephemerally at the edge to block malicious traffic.

## 2. Information You Submit (Audio Contributions)

When you upload an audio recording and metadata to RainAI, you are contributing to an open dataset used to train artificial intelligence models.

**By submitting a file, you agree that:**
1. **You have the right to provide this audio.** You are either the original creator of the audio, or the audio is in the public domain, or you have explicit permission to sub-license it under the license you select.
2. **Metadata is public:** The tags, descriptions, and the author/contributor name you provide will be stored in our database and distributed publicly alongside the audio file to provide attribution (XAI). Do not include personal or sensitive information in your tags or descriptions.
3. **Audio is utilized for AI Training:** The audio will be analyzed, processed, and used to train the RainAI neural network models. 

## 3. Data Retention and Deletion

* **Infrastructure Data:** Ephemeral IP hashes and rate-limit counters are automatically purged by Cloudflare typically within 24 hours.
* **Contribution Data:** Accepted audio contributions and metadata are merged into the permanent, open-source RainAI dataset repository. Because this data is distributed publicly via Git and used in compiled AI model weights, it may be technically impossible to completely erase your contribution once it has been integrated into a release. 

## 4. Cookies and Tracking

**We do not use advertising or tracking cookies.** 
Our application stores data locally on your device (using standard browser caches, Service Workers, or IndexedDB) exclusively to cache the WebAssembly application and offline attribution dictionaries to save your bandwidth and allow offline usage. 

## 5. Contact Us

If you have questions about this policy or need to report a copyright violation (DMCA takedown), please open an issue on our [GitHub Repository](https://github.com/spodeian/RainAI) or contact the maintainers.
