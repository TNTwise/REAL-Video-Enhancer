import { Box, Center, Flex, Heading, Image, SimpleGrid, Text, VStack } from "@chakra-ui/react";
import RVELogo from '/src/assets/logo-v2.svg';

const appVersion = import.meta.env.VITE_APP_VERSION ?? "0.1.0";

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <Box
      bg="var(--primary-widget)"
      borderColor="#343b47"
      borderWidth="1px"
      borderRadius="var(--border-radius)"
      p={4}
    >
      <Text fontSize="sm" color="#838ea2" mb={1}>{label}</Text>
      <Text fontWeight="semibold" color="#fff">{value}</Text>
    </Box>
  );
}

export function HomePage() {
  return (
    <Center h="full">
      <VStack align="stretch" maxW="md" w="full" gap={6}>
        {/* App Header */}
        <Flex direction="column" align="center" mb={2}>
          <Image src={RVELogo} w="100px" h="100px" mb={4} />
          <Heading size="lg">REAL Video Enhancer</Heading>
          <Text color="#838ea2" fontSize="sm">v{appVersion}</Text>
        </Flex>

        {/* Software Info */}
        <VStack align="stretch" gap={3}>
          <Heading size="sm">Software Information</Heading>
          <SimpleGrid columns={2} gap={4}>
            <InfoCard label="Python Version" value="3.12" />
            <InfoCard label="OpenCV Version" value="4.10.0" />
            <InfoCard label="PyTorch Version" value="2.5.1+cu124" />
            <InfoCard label="CUDA Available" value="Checking..." />
          </SimpleGrid>
        </VStack>

        {/* System Info */}
        <VStack align="stretch" gap={3}>
          <Heading size="sm">System Information</Heading>
          <SimpleGrid columns={2} gap={4}>
            <InfoCard label="OS" value={navigator.platform} />
            <InfoCard label="CPU" value={`${navigator.hardwareConcurrency ?? 0} cores`} />
            <InfoCard label="Device Memory" value={`${navigator.deviceMemory ?? 8} GB`} />
            <InfoCard label="Language" value={navigator.language} />
          </SimpleGrid>
        </VStack>

        {/* Footer */}
        <Text fontSize="xs" color="#838ea2" textAlign="center">
          Built with PyTorch, OpenCV, and Tauri
        </Text>
      </VStack>
    </Center>
  );
}
