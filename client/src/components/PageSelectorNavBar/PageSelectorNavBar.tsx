import { HStack, Link as ChakraLink } from "@chakra-ui/react";
import { Link as RouterLink } from "react-router-dom"; // 1. Import React Router's Link
import { primaryBGColor } from "@/styles/styles";
const links = [
  { name: "Home", href: "/home" },
  { name: "Download", href: "/download" },
];

export const Navbar = () => {
  return (
    <HStack gap={8} align="center" bg={primaryBGColor}>
      {links.map((link) => (
        <ChakraLink
          key={link.name}
          as={RouterLink}
          to={link.href} // ignore
          fontWeight="medium"
          color="blue.600"
          _hover={{
            color: "blue.500",
            textDecoration: "underline",
          }}
          transition="color 0.2s ease"
        >
          {link.name}
        </ChakraLink>
      ))}
    </HStack>
  );
};
