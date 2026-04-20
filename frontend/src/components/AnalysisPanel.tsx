import React, { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX, Loader2 } from "lucide-react";
import { API_ENDPOINTS } from "@/config/config";

interface AnalysisResult {
  timestamp?: string;
  driver_state: string | null;
  road_conditions: string | null;
  combined_context: string;
  frame_ids: Record<string, string>;
}

interface AnalysisPanelProps {
  loading?: boolean;
  currentAnalysis?: AnalysisResult;
}

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  loading = false,
  currentAnalysis
}) => {
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [activeTab, setActiveTab] = useState("live");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsLoading, setTtsLoading] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Add to history when new analysis comes in
  React.useEffect(() => {
    if (currentAnalysis) {
      setHistory(prev => [currentAnalysis, ...prev].slice(0, 10));
    }
  }, [currentAnalysis]);

  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
      audioRef.current = null;
    }
    setIsSpeaking(false);
  };

  const speakAnalysis = async (text: string) => {
    if (isSpeaking) {
      stopSpeaking();
      return;
    }

    if (!text.trim()) return;

    setTtsLoading(true);
    try {
      const response = await fetch(API_ENDPOINTS.tts, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || "TTS request failed");
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      stopSpeaking(); // clean up any previous audio
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      audio.onerror = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };

      setIsSpeaking(true);
      await audio.play();
    } catch (err) {
      console.error("Murf TTS error:", err);
      alert(`TTS Error: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setTtsLoading(false);
    }
  };

  const renderSpeakButton = (text: string) => (
    <Button
      size="sm"
      variant={isSpeaking ? "destructive" : "outline"}
      className="ml-auto flex items-center gap-1.5 text-xs"
      onClick={() => speakAnalysis(text)}
      disabled={ttsLoading || !text.trim()}
      id="murf-speak-btn"
    >
      {ttsLoading ? (
        <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Generating…</>
      ) : isSpeaking ? (
        <><VolumeX className="w-3.5 h-3.5" /> Stop</>
      ) : (
        <><Volume2 className="w-3.5 h-3.5" /> Speak</>
      )}
    </Button>
  );

  const renderAnalysis = (analysis: AnalysisResult, showSpeak = false) => (
    <div className="space-y-4">
      {showSpeak && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">
            Powered by Murf Falcon TTS
          </span>
          {renderSpeakButton(analysis.combined_context)}
        </div>
      )}

      <div>
        <h3 className="font-medium mb-2">Driver Status</h3>
        <p className="text-sm text-gray-600">
          {analysis.driver_state || "No driver analysis available"}
        </p>
      </div>

      <div>
        <h3 className="font-medium mb-2">Road Conditions</h3>
        <p className="text-sm text-gray-600">
          {analysis.road_conditions || "No road analysis available"}
        </p>
      </div>

      <div>
        <h3 className="font-medium mb-2">Combined Analysis</h3>
        <p className="text-sm text-gray-600">
          {analysis.combined_context || "No combined analysis available"}
        </p>
      </div>

      <div className="text-xs text-gray-400">
        {analysis.timestamp ?
          new Date(analysis.timestamp).toLocaleString() :
          'Timestamp not available'
        }
      </div>
    </div>
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="live">Live Analysis</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="live">
            {loading ? (
              <div className="text-center py-4 text-gray-500">
                Analyzing frame…
              </div>
            ) : currentAnalysis ? (
              renderAnalysis(currentAnalysis, true)
            ) : (
              <div className="text-center py-4 text-gray-500">
                No analysis available. Start the camera to begin.
              </div>
            )}
          </TabsContent>

          <TabsContent value="history">
            <ScrollArea className="h-[500px] pr-4">
              {history.length > 0 ? (
                <div className="space-y-8">
                  {history.map((analysis, index) => (
                    <div
                      key={`${analysis.timestamp}-${index}`}
                      className={index !== 0 ? "pt-4 border-t" : ""}
                    >
                      {renderAnalysis(analysis)}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  No historical data available yet.
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AnalysisPanel;