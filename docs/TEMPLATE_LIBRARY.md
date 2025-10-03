# Harvester SDK Template Library

## Overview

The Harvester SDK now includes **31 professional templates** (27 main + 4 process_dir) covering both developer workflows and essential business use cases.

## 📊 Template Categories

### 🚀 **NEW: Business & Content Templates (8)**
*Essential templates for normal users - content creators, marketers, business professionals*

1. **blog_post.j2** - SEO-friendly blog post writing
   - Engaging headlines and structure
   - Natural keyword optimization
   - Content quality guidelines
   - Multiple content types supported

2. **email_draft.j2** - Professional email drafting
   - Request, reply, introduction, follow-up types
   - Tone calibration (formal to casual)
   - Clear structure and CTAs
   - Common pitfalls avoided

3. **data_analysis.j2** - CSV/data analysis & insights
   - Statistical analysis
   - Pattern identification
   - Business insights
   - Actionable recommendations

4. **meeting_summary.j2** - Meeting notes transformation
   - Action items with owners
   - Decision tracking
   - Key discussion points
   - Follow-up planning

5. **product_description.j2** - E-commerce product copy
   - Features → Benefits transformation
   - Emotional storytelling
   - SEO optimization
   - Category-specific guidance

6. **seo_content.j2** - Search-optimized content
   - Keyword research integration
   - E-E-A-T optimization
   - Featured snippet strategies
   - Technical SEO best practices

7. **text_summary.j2** - Universal text summarization
   - Multiple summary types (executive, academic, news)
   - Adjustable length and focus
   - Maintains accuracy
   - Content-type specific approaches

8. **translation.j2** - Language translation & localization
   - Cultural adaptation
   - Regional variants
   - Terminology consistency
   - Format localization

### 💻 Code Quality & Development (7)
*Existing templates for developers*

9. **code_forge.j2** - Elite code analysis and upgrade
10. **code_forge_exact.j2** - Precise code transformation
11. **architectural_review.j2** - Architecture assessment
12. **agnostic_purity.j2** - Framework-agnostic refactoring
13. **performance_optimization.j2** - Performance improvements
14. **documentation.j2** - Code documentation generation
15. **document_improved.j2** - Enhanced documentation

### 🔄 Code Transformation - process_dir (4)
*Directory-level code operations*

16. **clean_code.j2** - Clean code refactoring
17. **modernize_code.j2** - Code modernization
18. **comprehensive_docs.j2** - Full documentation
19. **test_generation.j2** - Test case generation

### 🎨 Image Generation (4)
*Creative and product image prompts*

20. **basic_image_generation.j2** - Basic image prompts
21. **enhanced_image_prompt.j2** - Enhanced image prompts
22. **creative_art_generation.j2** - Creative art generation
23. **product_photography.j2** - Product photography

### 🔧 Prompting & Enhancement (4)
*Meta-prompting and quality assurance*

24. **prompt_enhancement.j2** - Prompt improvement
25. **prompt_improver_general.j2** - General prompt enhancement
26. **prompt_improver_program.j2** - Program-specific prompts
27. **quality_guardian.j2** - Quality assurance prompts

### 🛠️ Utilities (3)
*Specialized utilities*

28. **pattern_extraction.j2** - Extract patterns
29. **generate_schema.j2** - Schema generation
30. **knowledge_query.j2** - Knowledge extraction
31. **MASTER-TEMPLATE.j2** - Template template

## 🎯 User Focus Shift

**Before**: 90% Developer / 10% General
**After**: 60% Developer / 40% Business & Content

## 📈 Use Cases Now Covered

### ✅ Content Creation
- Blog posts and articles
- Product descriptions
- SEO-optimized content
- Social media (via general templates)

### ✅ Business Communication
- Professional emails
- Meeting summaries
- Executive summaries (via text_summary)
- Reports (via data_analysis)

### ✅ Data & Analysis
- CSV/spreadsheet analysis
- Data visualization recommendations
- Trend identification
- Business insights

### ✅ Marketing
- SEO content optimization
- Product copywriting
- Content strategy
- Keyword integration

### ✅ Productivity
- Text summarization
- Meeting notes processing
- Email drafting
- Document analysis

### ✅ Localization
- Multi-language translation
- Cultural adaptation
- Regional variants
- Format localization

### ✅ Development (Existing)
- Code refactoring
- Architecture review
- Documentation generation
- Test creation
- Performance optimization

## 🚀 Usage Examples

### Content Marketing Workflow
```bash
# Write a blog post
harvester batch topics.csv --template blog_post --model gemini-2.5-pro

# Optimize for SEO
harvester batch draft_posts.csv --template seo_content --model claude-sonnet-4-5

# Create product descriptions
harvester batch products.csv --template product_description --model gpt-5
```

### Business Productivity Workflow
```bash
# Summarize meeting transcripts
harvester batch meetings.csv --template meeting_summary --model gemini-2.5-flash

# Draft email responses
harvester batch email_contexts.csv --template email_draft --model deepseek-chat

# Analyze sales data
harvester batch sales_data.csv --template data_analysis --model claude-4
```

### Developer Workflow (Existing)
```bash
# Refactor codebase
harvester process ./src --template clean_code --model deepseek-reasoner

# Generate documentation
harvester process ./lib --template documentation --model gemini-2.5-pro

# Architecture review
harvester batch files.csv --template architectural_review --model gpt-5
```

### Localization Workflow
```bash
# Translate content to multiple languages
harvester batch content.csv --template translation --model gemini-2.5-pro

# Adapt marketing copy
harvester batch copy.csv --template translation --model claude-4
```

## 💡 Template Features

All templates include:
- ✅ **Clear Instructions** - Detailed guidance for AI models
- ✅ **Best Practices** - Industry-standard approaches
- ✅ **Quality Checklists** - Ensure excellent output
- ✅ **Multiple Formats** - Adaptable to different needs
- ✅ **Examples** - Real-world usage scenarios
- ✅ **Pitfall Avoidance** - Common mistakes to avoid
- ✅ **Professional Output** - Production-ready results

## 🎓 Template Quality Standards

Each template follows the Phoenix Methodology:
- **Clarity**: Clear, actionable instructions
- **Comprehensiveness**: Covers edge cases and variations
- **Context**: Provides background and rationale
- **Consistency**: Maintains SDK quality standards
- **Completeness**: Includes all necessary components

## 📚 Documentation

For detailed template usage:
- See individual `.j2` files for full documentation
- Each template includes inline examples
- Quality checklists ensure optimal results
- Best practices from industry experts

## 🌟 Impact

This expanded template library makes Harvester SDK valuable for:
- **Content Creators** - Blog posts, SEO, product copy
- **Marketers** - SEO optimization, product descriptions
- **Business Users** - Emails, meetings, data analysis
- **Translators** - Professional localization
- **Developers** - Code quality and documentation (existing)
- **Data Analysts** - Insights and reporting
- **Product Managers** - Documentation and communication

---

**© 2025 Quantum Encoding Ltd**
Open Source - MIT License
All templates available for unrestricted use
