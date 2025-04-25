declare module 'wink-lemmatizer' {
  interface Lemmatizer {
    lemmatize(word: string): string;
  }
  const lemmatizer: Lemmatizer;
  export default lemmatizer;
}

declare module 'wink-lemmatizer' {
  export function noun(word: string): string;
  export function verb(word: string): string;
  export function adjective(word: string): string;
  export function adverb(word: string): string;
}
